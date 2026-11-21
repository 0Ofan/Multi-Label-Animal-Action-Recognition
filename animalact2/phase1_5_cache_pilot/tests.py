"""Integrity tests use separate fixtures; never damage the real pilot cache."""
from __future__ import annotations
import copy
import json
import shutil
import unittest
import numpy as np
import torch
from .cache import CACHE,select_plan,selected_records,context,frame_key,clip_descriptor,FrameStore,read_array,CacheIntegrityError
from .dataset import CachedFeatureDataset,collate_features
from .support import OUT,digest


class CacheTests(unittest.TestCase):
    def test_key_changes_with_video_frame_preprocessing_or_weights(self):
        ctx=context();r=selected_records(select_plan())[0];key=frame_key(r,r["frame_indices"][0],ctx)
        other=copy.deepcopy(r);other["video_sha256"]="0"*64
        self.assertNotEqual(key,frame_key(other,r["frame_indices"][0],ctx))
        self.assertNotEqual(key,frame_key(r,r["frame_indices"][0]+1,ctx))
        for field,value in (("weights_SHA256","f"*64),("preprocessing",dict(resize_short_edge=225))):
            changed=copy.deepcopy(ctx);changed[field]=value
            self.assertNotEqual(key,frame_key(r,r["frame_indices"][0],changed))

    def test_corrupt_array_is_rejected(self):
        source=next((CACHE/"frames").glob("shard_*"));target=OUT/"test_fixtures/corrupted_shard"
        if not target.exists():shutil.copytree(source,target)
        p=target/"features.npy"
        with p.open("r+b") as f:f.seek(0);f.write(b"BAD!!")
        with self.assertRaises(CacheIntegrityError):read_array(target)

    def test_wrong_receipt_or_context_is_rejected(self):
        source=next((CACHE/"frames").glob("shard_*"))
        with self.assertRaises(CacheIntegrityError):read_array(source,dict(context_SHA256="wrong"))
        ctx=context();ctx["weights_SHA256"]="0"*64
        with self.assertRaises(CacheIntegrityError):FrameStore(CACHE,ctx)

    def test_every_clip_fp32_length_label_and_source_alignment(self):
        for split,n in (("train",100),("valid",20)):
            d=CachedFeatureDataset(split);self.assertEqual(len(d),n)
            for i in range(len(d)):
                sample=d[i];r=d.records[i]
                self.assertEqual(tuple(sample["features"].shape),(r["effective_length"],512))
                self.assertEqual(sample["features"].dtype,torch.float32)
                self.assertEqual(sample["target"].tolist(),r["target_119"])
                self.assertEqual(sample["frame_indices"].tolist(),r["frame_indices"])
                self.assertEqual(sample["label_sources"],r["label_evidence"])

    def test_short_feature_batch_has_no_forced_16_and_padding_is_storage_only(self):
        d=CachedFeatureDataset("train")
        chosen=[d[i] for length in (1,2,3) for i,r in enumerate(d.records) if r["effective_length"]==length][:3]
        b=collate_features(chosen)
        self.assertLessEqual(b["frames"].shape[1],3)
        self.assertTrue(torch.equal(b["valid_frame_mask"].sum(1),b["lengths"]))
        self.assertTrue(torch.all(b["frames"][b["padding_mask"]]==0))
        self.assertTrue(torch.all(b["frame_indices"][b["padding_mask"]]==-1))

    def test_no_test_or_scope_expansion(self):
        with self.assertRaises(ValueError):CachedFeatureDataset("test")
        with self.assertRaises(ValueError):CachedFeatureDataset("train",clip_ids=["not_in_locked_pilot"])

    def test_uncommitted_partial_shard_is_ignored_on_resume(self):
        from .support import write_json
        root=OUT/"test_fixtures/interrupted_commit"
        write_json(root/"context.json",context())
        source=next((CACHE/"frames").glob("shard_*"));target=root/"frames"/source.name
        if not target.exists():shutil.copytree(source,target)
        pending=root/"frames/.pending_simulated_interruption";pending.mkdir(exist_ok=True)
        (pending/"features.npy").write_bytes(b"uncommitted partial write")
        store=FrameStore(root,context())
        _,receipt=read_array(source)
        self.assertEqual(len(store.locations),len(receipt["frames"]))


if __name__=="__main__":unittest.main(verbosity=2)
