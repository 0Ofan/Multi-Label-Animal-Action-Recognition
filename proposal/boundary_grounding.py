"""Dense ResNet features and query-conditioned activity/start/end supervision.

Train/Valid development only. Historical classification and Test artifacts are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from .grounding import read_queries, temporal_iou
from .predict import ProposalPredictor


@torch.inference_mode()
def dense_features(predictor, path, rate=4., batch_size=32):
    if not np.isfinite(rate) or rate <= 0 or batch_size < 1:
        raise ValueError("Positive sampling rate and batch size required")
    capture = cv2.VideoCapture(str(path))
    pending, features, times = [], [], []
    previous = -1.
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0.
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid video duration")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise ValueError("Decoder timestamps are not strictly increasing")
            previous = timestamp
            if timestamp + 1e-6 < len(times) / rate:
                continue
            if timestamp >= duration:
                raise ValueError("Decoder timestamp exceeds video duration")
            times.append(timestamp)
            pending.append(predictor.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            if len(pending) == batch_size:
                features.append(predictor.encoder(torch.stack(pending).to(predictor.device)).cpu())
                pending.clear()
        if pending:
            features.append(predictor.encoder(torch.stack(pending).to(predictor.device)).cpu())
        if not times or previous < duration - max(2 / fps, .1):
            raise ValueError("Video decode is empty or incomplete")
    finally:
        capture.release()
    timestamps = np.asarray(times)
    edges = np.r_[0., (timestamps[:-1] + timestamps[1:]) / 2, duration]
    return dict(features=torch.cat(features), timestamps=timestamps.tolist(), edges=edges.tolist(),
                duration=duration, rate=rate)


def targets(references, edges):
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Expected increasing finite time edges")
    target = np.zeros((len(edges) - 1, 3), dtype=np.float32)
    merged = []
    for start, end in sorted(references):
        if not np.isfinite([start, end]).all() or not edges[0] <= start < end <= edges[-1] + 1e-5:
            raise ValueError("Reference interval is outside video")
        end = min(end, edges[-1])
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    for start, end in merged:
        target[:, 0] += np.maximum(0., np.minimum(edges[1:], end) - np.maximum(edges[:-1], start)) / np.diff(edges)
        target[np.argmin(abs(edges[:-1] - start)), 1] = 1.
        target[np.argmin(abs(edges[1:] - end)), 2] = 1.
    target[:, 0] = np.clip(target[:, 0], 0, 1)
    return torch.from_numpy(target)


class BoundaryModel(nn.Module):
    def __init__(self, animals):
        super().__init__()
        self.animal = nn.Embedding(animals + 1, 16, padding_idx=0)
        self.action = nn.Embedding(119, 16)
        # Reuse the project's 512-dimensional ResNet -> bidirectional GRU method.
        self.temporal = nn.GRU(512 + 32, 128, batch_first=True, bidirectional=True)
        self.boundaries = nn.Linear(256, 3)
        self.presence = nn.Linear(256, 1)

    def forward(self, features, animal, action):
        query = torch.cat((self.animal(animal), self.action(action)), -1)
        sequence = torch.cat((features, query[:, None].expand(-1, features.shape[1], -1)), -1)
        encoded, _ = self.temporal(sequence)
        return self.boundaries(encoded), self.presence(encoded.mean(1)).squeeze(-1)


def boundary_loss(logits, presence, target):
    positives = target.sum((0, 1))
    weight = ((target.shape[0] * target.shape[1] - positives) / positives.clamp_min(1)).clamp(1, 20)
    return (nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
            + nn.functional.binary_cross_entropy_with_logits(presence, target[:, :, 0].amax(1).gt(0).float()))


def decode(logits, presence, edges, threshold=.5):
    scores = torch.as_tensor(logits).sigmoid().cpu().numpy()
    edges = np.asarray(edges)
    if scores.shape != (len(edges) - 1, 3) or np.any(np.diff(edges) <= 0):
        raise ValueError("Prediction and timestamp geometry mismatch")
    if torch.as_tensor(presence).sigmoid().item() < threshold:
        return []
    active = scores[:, 0] >= threshold
    changes = np.diff(np.r_[False, active, False].astype(int))
    intervals = []
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start_options = np.arange(max(0, first - 1), min(len(active), first + 2))
        end_options = np.arange(max(0, last - 2), min(len(active), last + 1))
        start = int(start_options[np.argmax(scores[start_options, 1])])
        end = int(end_options[np.argmax(scores[end_options, 2])]) + 1
        if end <= start:
            start, end = first, last
        intervals.append(dict(interval=[float(edges[start]), float(edges[end])],
                              score=float(scores[first:last, 0].mean())))
    return sorted(intervals, key=lambda row: -row['score'])


def evaluate_record(query, intervals):
    references = query['references']
    if not references:
        return dict(positive=False, false_positive=bool(intervals))
    if not intervals:
        return dict(positive=True, iou=0., boundary_hit=False, missed=True,
                    start_error=None, end_error=None)
    predicted = np.asarray([intervals[0]['interval']])
    overlaps = [temporal_iou(predicted, [reference])[0] for reference in references]
    index = int(np.argmax(overlaps))
    start, end = abs(predicted[0] - np.asarray(references[index]))
    return dict(positive=True, iou=float(overlaps[index]), boundary_hit=bool(max(start, end) <= .5),
                missed=False, start_error=float(start), end_error=float(end))


def summary(records):
    positives = [r for r in records if r['positive']]
    negatives = [r for r in records if not r['positive']]
    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    return dict(positive_queries=len(positives), negative_queries=len(negatives),
                miou=mean(positives, 'iou'), boundary_hit_at_0_5s=mean(positives, 'boundary_hit'),
                r1_iou_05=mean([dict(hit=r['iou'] >= .5) for r in positives], 'hit'),
                r1_iou_07=mean([dict(hit=r['iou'] >= .7) for r in positives], 'hit'),
                miss_rate=mean(positives, 'missed'), negative_false_positive_rate=mean(negatives, 'false_positive'))


def dataset(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    ids = {split: sorted(r['video_id'] for r in manifest['samples'] if r['split'] == split)
           for split in ('train', 'valid')}
    classes = [r['name'] for r in sorted(json.loads((root / 'action_classes.json').read_text(encoding='utf-8'))['classes'], key=lambda r: r['class_id'])]
    class Reader:
        def read(self, name):
            return (root / name.removeprefix('AnimalAct/')).read_bytes()
    queries, animals = read_queries(Reader(), ids, classes)
    for split, rows in queries.items():
        for record in manifest['samples']:
            if record['split'] == split:
                raw = (root / record['annotation_path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != record['annotation_sha256']:
                    raise ValueError('Annotation checksum mismatch')
        augmented = []
        for row in rows:
            present = {q['action'] for q in rows if q['video_id'] == row['video_id'] and q['animal'] == row['animal']}
            candidates = [a for a in range(119) if a not in present]
            seed = int(hashlib.sha256((row['video_id'] + str(row['action'])).encode()).hexdigest()[:8], 16)
            absent = candidates[seed % len(candidates)] if candidates else None
            if absent is not None:
                # Unannotated pairs are assumed negatives, not independently verified absence.
                augmented.append(dict(row, action=absent, references=[], query=f"A {row['animal'].lower()} is {classes[absent].lower()}."))
        queries[split] = rows + augmented
    return manifest, queries, animals, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'extract', 'train'), required=True)
    parser.add_argument('--rate', type=float, default=4.)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0 or args.epochs < 1:
        parser.error('Positive rate and epochs required')
    torch.set_num_threads(4)
    torch.manual_seed(6116)
    root = args.data.resolve()
    manifest, queries, animals, classes = dataset(root)
    rows = [r for r in manifest['samples'] if r['split'] in queries]
    if args.mode == 'smoke':
        rows = [next(r for r in rows if r['split'] == split) for split in queries]
        queries = {split: [q for q in qs if q['video_id'] in {r['video_id'] for r in rows}] for split, qs in queries.items()}
    output = args.output.resolve()
    if (output / 'boundary.pt').exists():
        raise FileExistsError('Use a new output directory; existing checkpoints are protected')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'features'
    cache.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = ProposalPredictor(device) if args.mode != 'train' else None
    for index, row in enumerate(rows):
        path = cache / (row['video_id'] + '.pt')
        source = root / row['video_path']
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.exists():
            value = torch.load(path, weights_only=True, map_location='cpu')
            if value['rate'] != args.rate or value['source_sha256'] != digest:
                raise ValueError('Cache/source settings changed')
        elif predictor is None:
            raise FileNotFoundError(f'Extract features before training: {path}')
        else:
            value = dense_features(predictor, source, args.rate)
            value['source_sha256'] = digest
            torch.save(value, path)
        if abs(value['duration'] - row['duration']) > max(1 / row['fps'], .05):
            raise ValueError('Annotation/video duration mismatch')
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{len(rows)} videos', flush=True)
    if args.mode == 'extract':
        print(json.dumps(dict(videos=len(rows), device=device, test_read=False)))
        return
    model = BoundaryModel(len(animals)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    def example(query):
        value = torch.load(cache / (query['video_id'] + '.pt'), weights_only=True, map_location='cpu')
        return value, value['features'][None].to(device), torch.tensor([query['animal_id']], device=device), torch.tensor([query['action']], device=device)
    history, best = [], (-1., -1., -1.)
    for epoch in range(1, (1 if args.mode == 'smoke' else args.epochs) + 1):
        model.train()
        for i in torch.randperm(len(queries['train'])).tolist():
            q = queries['train'][i]
            value, x, animal, action = example(q)
            logits, presence = model(x, animal, action)
            target = targets(q['references'], value['edges'])[None].to(device)
            loss = boundary_loss(logits, presence, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            records = []
            for q in queries['valid']:
                value, x, animal, action = example(q)
                logits, presence = model(x, animal, action)
                intervals = decode(logits[0], presence[0], value['edges'])
                baseline = evaluate_record(q, [dict(interval=[0., q['duration']])])
                records.append(dict(video_id=q['video_id'], query=q['query'], references=q['references'],
                                    baseline=baseline, intervals=intervals, **evaluate_record(q, intervals)))
        metrics = summary(records)
        metrics['whole_video_baseline'] = summary([r['baseline'] for r in records])
        history.append(dict(epoch=epoch, validation=metrics))
        print(json.dumps(history[-1]), flush=True)
        selection = (metrics['boundary_hit_at_0_5s'], metrics['miou'], -metrics['negative_false_positive_rate'])
        if selection > best:
            best = selection
            torch.save(dict(state_dict=model.state_dict(), animals=animals, classes=classes, rate=args.rate,
                            epoch=epoch, selection_split='valid', threshold=.5, schema='boundary-v1',
                            validation=metrics, smoke=args.mode == 'smoke'), output / 'boundary.pt')
            (output / 'validation_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (output / 'results.json').write_text(json.dumps(dict(history=history, test_read=False,
        smoke=args.mode == 'smoke', selection_metric='boundary_hit_at_0_5s',
        negative_policy='unannotated animal/action pairs assumed absent'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
