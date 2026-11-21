"""Fixed small Train/Valid diagnostics. No epoch loop, checkpoints or feature cache."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import ReleaseClipDataset, RELEASE_ID, PROJECT_ROOT
from .collate import collate_clips, move_batch_to_device
from .video_decoder import decode_exact_frames
from .tests import test_contracts

OUT = Path(__file__).resolve().parent
READS = set()


def runtime_guard(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Network use is not part of this diagnostic")
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    if write and not p.is_relative_to(OUT):
        raise RuntimeError("Diagnostic writes must stay in phase1_5_1")
    if p.is_relative_to(PROJECT_ROOT):
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        low = "/" + rel.casefold()
        if "/videos/test/" in low or "/annotations/test/" in low or "/test_metadata_only/" in low or "predictions" in low or "model5" in low or "model6" in low:
            raise RuntimeError("Forbidden diagnostic source: " + rel)
        if p.suffix.lower() in (".pth", ".pt", ".safetensors", ".npy", ".npz") and rel != ".cache/torch/checkpoints/resnet18-f37072fd.pth":
            raise RuntimeError("Old task checkpoint/feature cache use blocked: " + rel)
        if not write:
            READS.add(rel)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def choose_samples(dataset):
    selected = []
    for lo, hi in ((1, 3), (4, 15), (16, 16)):
        pool = [r for r in dataset.records if lo <= r["effective_length"] <= hi]
        if lo == 1:
            pool.sort(key=lambda r: (r["effective_length"] != 1, r["parent_video_id"] != "CLZLJTLI", r["clip_id"]))
        elif lo == 4:
            pool.sort(key=lambda r: (r["effective_length"] != 10, r["parent_video_id"] != "BIFKWBQT", r["clip_id"]))
        else:
            pool.sort(key=lambda r: (len({a["animal_name"] for e in r["label_evidence"] for a in e["contributors"]}) < 2, len(r["class_ids"]) < 2, r["clip_id"]))
        selected.append(pool[0]["clip_id"])
    return ReleaseClipDataset(dataset.split, clip_ids=selected)


def check_reference_decode(dataset):
    records = []
    for record in dataset.records:
        path = PROJECT_ROOT / record["video_path"]
        assert sha(path) == record["video_sha256"]
        assert sha(PROJECT_ROOT / record["annotation_path"]) == record["annotation_sha256"]
        decoded = decode_exact_frames(path, record["frame_indices"], expected_num_frames=record["parent_num_frames"], expected_fps=record["fps"])
        # Independent reference: full read() traversal, no seeking/grab-retrieve API.
        capture = cv2.VideoCapture(str(path))
        try:
            expected, k = {}, 0
            while k <= record["frame_indices"][-1]:
                ok, bgr = capture.read()
                assert ok and bgr is not None
                if k in record["frame_indices"]:
                    expected[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                k += 1
        finally:
            capture.release()
        assert all(np.array_equal(rgb, expected[k]) for rgb, k in zip(decoded.rgb, decoded.frame_indices))
        records.append(dict(clip_id=record["clip_id"], parent_video_id=record["parent_video_id"], split=record["split"],
                            effective_length=record["effective_length"], available_strict_frames=record["available_strict_frame_count"],
                            frame_indices=record["frame_indices"], target_class_ids=record["class_ids"], target_action_names=record["action_names"],
                            annotation_uids=record["all_positive_label_source_uids"], annotation_refs=sorted({a["annotation_ref"] for e in record["label_evidence"] for a in e["contributors"]}),
                            source_video_SHA_verified=True, source_annotation_SHA_verified=True, reference_RGB_pixels_equal=True,
                            decoded_width=decoded.width, decoded_height=decoded.height, timestamps_seconds=list(decoded.timestamps_seconds),
                            maximum_timestamp_error_seconds=max(abs(k / decoded.fps - t) for k, t in zip(decoded.frame_indices, decoded.timestamps_seconds))))
    return records


def gradient_summary(model):
    result = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            norm = float(p.grad.detach().norm())
            assert norm > 0, name
            result[name] = norm
        else:
            assert p.grad is None, name
    return result


def inputs(batch):
    return batch["frames"], batch["lengths"], batch["valid_frame_mask"]


def main():
    started = time.perf_counter()
    sys.addaudithook(runtime_guard)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="New result directory inside phase1_5_1; existing results are never overwritten")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(OUT):
        raise ValueError("Diagnostic results must stay in the independent phase directory")
    if (output_dir / "validation_results.json").exists() or (output_dir / "unit_test_results.txt").exists():
        raise FileExistsError("Choose a new --output-dir to preserve the completed diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_contracts)
    stream = io.StringIO()
    unit_start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output_dir / "unit_test_results.txt").write_text(stream.getvalue(), encoding="utf-8")
    if not result.wasSuccessful():
        print(stream.getvalue(), flush=True)
        raise RuntimeError("Unit tests failed; no real diagnostic update started")
    unit_seconds = time.perf_counter()-unit_start
    print(json.dumps(dict(unit_tests=result.testsRun, failures=0, errors=0, seconds=unit_seconds)), flush=True)
    datasets = {s: choose_samples(ReleaseClipDataset(s)) for s in ("train", "valid")}
    decode_start = time.perf_counter()
    samples = [r for d in datasets.values() for r in check_reference_decode(d)]
    batches = {s: next(iter(DataLoader(d, batch_size=3, shuffle=False, num_workers=0, collate_fn=collate_clips))) for s, d in datasets.items()}
    decode_seconds = time.perf_counter()-decode_start
    assert len(samples) == 6 and {r["split"] for r in samples} == {"train", "valid"}
    assert len({r["parent_video_id"] for r in samples}) == 6
    for s, batch in batches.items():
        assert torch.equal(batch["valid_frame_mask"].sum(dim=1), batch["lengths"])
        assert batch["frames"].shape[1] == int(batch["lengths"].max())
        assert all(m["target_119"] == batch["target"][i].tolist() for i, m in enumerate(batch["metadata"]))
        print(json.dumps(dict(split=s, clip_ids=batch["clip_ids"], lengths=batch["lengths"].tolist(), frames=list(batch["frames"].shape), real_frames=int(batch["lengths"].sum())), ensure_ascii=False), flush=True)
    models = test_contracts.ModelContractTests.models
    pretrained_states = [state_sha(m.backbone.encoder) for m in models]
    assert len(set(pretrained_states)) == 1
    model_results, cpu_feature_hashes = [], []
    for model in models:
        name = type(model).__name__
        model.cpu().train()
        model.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        observed_real_frames = []
        hook = model.backbone.encoder.register_forward_pre_hook(lambda _, args: observed_real_frames.append(len(args[0])))
        before_BN = {n: p.clone() for n, p in model.backbone.encoder.named_buffers()}
        cpu_out = model(*inputs(batches["train"]), return_aux=True)
        hook.remove()
        cpu_logits = cpu_out["logits"].detach()
        assert list(cpu_logits.shape) == [3, 119] and torch.isfinite(cpu_logits).all()
        assert sum(observed_real_frames) == int(batches["train"]["lengths"].sum())
        assert max(observed_real_frames) <= model.backbone.frame_chunk_size
        assert all(torch.equal(p, before_BN[n]) for n, p in model.backbone.encoder.named_buffers())
        loss_cpu = nn.functional.binary_cross_entropy_with_logits(cpu_out["logits"], batches["train"]["target"])
        loss_cpu.backward()
        cpu_gradients = gradient_summary(model)
        cpu_feature_hashes.append(tensor_sha(cpu_out["features"]))
        if name == "Model1":
            poisoned = batches["train"]["frames"].clone()
            poisoned[~batches["train"]["valid_frame_mask"]] = float("nan")
            with torch.no_grad():
                poison_out = model(poisoned, batches["train"]["lengths"], batches["train"]["valid_frame_mask"])
            torch.testing.assert_close(poison_out, cpu_logits, atol=1e-6, rtol=1e-5)
        model.eval()
        with torch.no_grad():
            valid_out = model(*inputs(batches["valid"]), return_aux=True)
        assert list(valid_out["logits"].shape) == [3, 119] and torch.isfinite(valid_out["logits"]).all()
        cpu_seconds = time.perf_counter()-begin
        cpu_attn_check = None
        if name == "Model3":
            w = valid_out["temporal_attention"]
            torch.testing.assert_close(w.sum(-1), torch.ones(3, 119), atol=1e-6, rtol=0)
            assert torch.all(w.masked_select(batches["valid"]["padding_mask"][:, None, :].expand_as(w)) == 0)
            cpu_attn_check = dict(shape=list(w.shape), masked_mass=0., time_sum_max_error=float((w.sum(-1)-1).abs().max()))
        model.zero_grad(set_to_none=True)
        del cpu_out, valid_out
        print(json.dumps(dict(model=name, CPU="forward/backward passed", train_BCE=float(loss_cpu.detach()), seconds=cpu_seconds)), flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_start = time.perf_counter()
        model.to("cuda").train()
        train_gpu = move_batch_to_device(batches["train"], "cuda")
        valid_gpu = move_batch_to_device(batches["valid"], "cuda")
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        out_gpu = model(*inputs(train_gpu), return_aux=True)
        assert out_gpu["logits"].shape == (3, 119) and torch.isfinite(out_gpu["logits"]).all()
        torch.testing.assert_close(out_gpu["logits"].detach().cpu(), cpu_logits, atol=2e-4, rtol=2e-4)
        cpu_cuda_max_error = float((out_gpu["logits"].detach().cpu()-cpu_logits).abs().max())
        loss_gpu = nn.functional.binary_cross_entropy_with_logits(out_gpu["logits"], train_gpu["target"])
        loss_gpu.backward()
        gpu_gradients = gradient_summary(model)
        optimizer.step()  # Exactly one diagnostic SGD update per model, Train only.
        updates = {n: float((p.detach()-before[n]).norm()) for n, p in model.named_parameters() if p.requires_grad}
        assert all(v > 0 for v in updates.values())
        model.eval()
        with torch.no_grad():
            valid_logits_gpu = model(*inputs(valid_gpu))
        assert valid_logits_gpu.shape == (3, 119) and torch.isfinite(valid_logits_gpu).all()
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter()-gpu_start
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        assert peak_allocated < torch.cuda.get_device_properties(0).total_memory
        model.zero_grad(set_to_none=True)
        model.cpu()
        assert state_sha(model.backbone.encoder) == pretrained_states[0]
        model_results.append(dict(model=name, total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  frozen_backbone_parameters=sum(p.numel() for p in model.backbone.parameters()), official_backbone_SHA256=model.backbone.weight_info["sha256"], backbone_state_SHA256=pretrained_states[0],
                                  initial_head_seed=42, CPU_output_shape=[3,119], CUDA_output_shape=[3,119], outputs_finite=True,
                                  CPU_train_BCE_diagnostic_only=float(loss_cpu.detach()), CUDA_train_BCE_diagnostic_only=float(loss_gpu.detach().cpu()),
                                  CPU_gradient_norms=cpu_gradients, CUDA_gradient_norms=gpu_gradients, diagnostic_update_L2_per_parameter=updates, optimizer="SGD diagnostic only", learning_rate=1e-3,
                                  diagnostic_optimizer_steps=1, updated_split="train", Valid_gradients_or_updates=False,
                                  CPU_seconds=cpu_seconds, CUDA_seconds=gpu_seconds, CUDA_peak_allocated_bytes=peak_allocated, CUDA_peak_reserved_bytes=peak_reserved,
                                  CPU_CUDA_logits_max_absolute_error=cpu_cuda_max_error, valid_frames_seen_by_CNN=sum(observed_real_frames), CNN_frame_chunk_sizes=observed_real_frames,
                                  padded_frames_seen_by_CNN=0, frozen_backbone_and_BN_unchanged=True, attention_check=cpu_attn_check))
        print(json.dumps(dict(model=name, CUDA="forward/backward + one Train update passed", peak_allocated_MiB=peak_allocated/2**20, peak_reserved_MiB=peak_reserved/2**20, seconds=gpu_seconds)), flush=True)
        del train_gpu, valid_gpu, out_gpu, valid_logits_gpu, before, optimizer, loss_cpu, loss_gpu
        torch.cuda.empty_cache()
    assert len(set(cpu_feature_hashes)) == 1, "The three models did not receive identical frame features"
    result = dict(status="passed", release_id=RELEASE_ID, python=sys.version, executable=sys.executable,
                  torch_version=str(torch.__version__), torchvision_version=__import__("torchvision").__version__, opencv_version=cv2.__version__,
                  GPU=torch.cuda.get_device_name(0), GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory, diagnostic_precision="FP32; TF32 disabled",
                  default_backbone_frozen=True, unit_tests=result.testsRun, unit_failures=0, unit_errors=0, unit_seconds=unit_seconds,
                  real_sample_records=samples, real_sample_count=6, source_parent_count=6, decode_and_collate_seconds=decode_seconds,
                  batches={s: dict(frames_shape=list(b["frames"].shape), lengths=b["lengths"].tolist(), valid_frame_mask=b["valid_frame_mask"].tolist(), frame_indices=b["frame_indices"].tolist(), target_shape=list(b["target"].shape), target_SHA256=tensor_sha(b["target"])) for s,b in batches.items()},
                  models=model_results, same_initial_backbone=True, same_initial_Model2_Model3_BiGRU=True,
                  same_CPU_frame_features_SHA256=cpu_feature_hashes[0], real_padding_poison_test_passed=True,
                  parameters_persisted=False, feature_or_RGB_cache_written=False, prediction_files_written=False, trained_model_saved=False,
                  Test_data_used=False, old_weights_loaded=False, official_ImageNet_weights_reused_from_local_project=True,
                  formal_training=False, next_stage_started=False, total_validation_seconds=time.perf_counter()-started,
                  project_read_paths=sorted(READS))
    (output_dir / "validation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status="passed", unit_tests=result["unit_tests"], real_clips=6, models=3, maximum_peak_allocated_MiB=max(r["CUDA_peak_allocated_bytes"] for r in model_results)/2**20, total_seconds=result["total_validation_seconds"])), flush=True)


if __name__ == "__main__":
    main()
