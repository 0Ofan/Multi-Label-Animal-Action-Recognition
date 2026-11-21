"""Read-interface checks and isolated corruption fixtures; no updates or checkpoints."""
import copy
import json
import shutil
import unittest
import numpy as np
import torch
from .storage import CACHE,context,frame_key,read_array,commit_array,CacheIntegrityError
from .dataset import CachedFeatureDataset,collate_features
from .support import OUT,PILOT


class FullCacheTests(unittest.TestCase):
    def test_exact_Pilot_context_and_keys(self):
        ctx=context();old=json.loads((PILOT/"cache/context.json").read_text(encoding="utf-8"))
        self.assertEqual(ctx,old)
        d=CachedFeatureDataset("train");r=d.records[0];key=frame_key(r,r["frame_indices"][0],ctx)
        other=copy.deepcopy(ctx);other["weights_SHA256"]="0"*64
        self.assertNotEqual(key,frame_key(r,r["frame_indices"][0],other))

    def test_full_counts_and_Test_disabled(self):
        self.assertEqual(len(CachedFeatureDataset("train")),4667)
        self.assertEqual(len(CachedFeatureDataset("valid")),979)
        with self.assertRaises(ValueError):CachedFeatureDataset("test")

    def test_variable_length_batch_and_original_labels(self):
        d=CachedFeatureDataset("train")
        indices=[next(i for i,r in enumerate(d.records) if r["effective_length"]==n) for n in (1,2,3)]
        samples=[d[i] for i in indices];b=collate_features(samples)
        self.assertEqual(tuple(b["frames"].shape),(3,3,512))
        self.assertEqual(b["lengths"].tolist(),[1,2,3])
        self.assertTrue(torch.all(b["frames"][b["padding_mask"]]==0))
        for i,source_index in enumerate(indices):
            self.assertEqual(b["target"][i].tolist(),d.records[source_index]["target_119"])
            self.assertEqual(b["label_sources"][i],d.records[source_index]["label_evidence"])

    def test_corrupt_copy_is_rejected_without_touching_real_cache(self):
        source=next((CACHE/"frames").glob("shard_*"));target=OUT/"test_fixtures/corrupt_copy"
        if not target.exists():shutil.copytree(source,target)
        with (target/"features.npy").open("r+b") as f:f.write(b"BAD!!")
        with self.assertRaises(CacheIntegrityError):read_array(target)

    def test_atomic_commit_refuses_old_or_existing_targets(self):
        with self.assertRaises(ValueError):commit_array(PILOT/"must_not_be_written",np.zeros((1,512),dtype=np.float32),{})
        source=next((CACHE/"frames").glob("shard_*"))
        with self.assertRaises(FileExistsError):commit_array(source,np.zeros((1,512),dtype=np.float32),{})


if __name__=="__main__":unittest.main(verbosity=2)
