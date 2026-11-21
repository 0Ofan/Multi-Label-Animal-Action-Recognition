"""Portable runtime tests: no original dataset, training, or Test access."""
import unittest,json,os,tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
from animalact2.dashboard.registry import ROOT,model_info,classes
from animalact2.dashboard.video import sample_indices
from animalact2.dashboard.schema import validate_prediction
from animalact2.dashboard.evaluation_data import evaluation
from animalact2.phase2_2.heads import VideoMAEProposalHead
class PortableTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(model_info()["threshold"],.12)
        self.assertEqual(model_info("bigru_mean")["threshold"],.07)
        self.assertEqual(len(classes()),119)
    def test_sampling(self):
        for n in range(1,50):
            ids,_=sample_indices(n,24.)
            self.assertEqual(len(ids),min(n,16));self.assertEqual(ids,sorted(set(ids)))
    def test_evaluation(self):
        self.assertEqual(len(evaluation("summary")["rows"]),3)
        self.assertEqual(len(evaluation("runs")["rows"]),9)
        for model in ("model1","model2","model3"):
            for seed in (42,123,2026):self.assertEqual(len(evaluation("confusion",model,seed)["rows"]),31)
        with self.assertRaises(ValueError):evaluation("test")
    def test_masks(self):
        for kind in ("mean","bigru_mean","bigru_attention"):
            h=VideoMAEProposalHead(kind).eval();x=torch.randn(3,8,768);lens=torch.tensor([1,3,8]);mask=torch.arange(8)[None]<lens[:,None]
            before=h(x,lens,mask);x[~mask]=1e4;torch.testing.assert_close(before,h(x,lens,mask))
    @unittest.skipUnless(os.environ.get("RUN_MODEL_SMOKE")=="1","enable real pretrained CPU smoke explicitly")
    def test_real_encoder(self):
        from animalact2.dashboard.predictor import Predictor
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"synthetic_control.avi"
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),24,(64,36))
            for color in (30,80,150):writer.write(np.full((36,64,3),color,dtype=np.uint8))
            writer.release()
            predictor=Predictor("cpu")
            for key in ("attention","bigru_mean"):
                result=validate_prediction(predictor.predict(path,key))
                self.assertEqual(result["feature_shape"],[1,2,768]);self.assertEqual(result["video"]["sampled_frame_indices"],[0,1,2])
                self.assertFalse(result["class_scores"][23]["selected"])
            # Single-frame window remains one actual frame plus a zero alignment slot.
            result=predictor.predict(path,"attention",0,.01)
            self.assertEqual(result["feature_shape"],[1,1,768])
            full=Path(tmp)/"sixteen_real_frames.avi"
            writer=cv2.VideoWriter(str(full),cv2.VideoWriter_fourcc(*"MJPG"),24,(64,36))
            for i in range(16):writer.write(np.full((36,64,3),20+i*10,dtype=np.uint8))
            writer.release()
            result=predictor.predict(full,"attention")
            self.assertEqual(result["feature_shape"],[1,8,768])
            self.assertEqual(result["video"]["sampled_frame_indices"],list(range(16)))
            try:
                import resource
                print("Linux CPU smoke peak RSS/KiB:",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            except ImportError:pass
    def test_three_pages(self):
        from streamlit.testing.v1 import AppTest
        for page in ("video","results","evaluation"):
            app=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=90);app.query_params["page"]=page;app.run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertFalse(any("could not be loaded" in item.value for item in app.error))
if __name__=="__main__":unittest.main(verbosity=2)
