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
    def test_semantic_search_fallback(self):
        from animalact2.dashboard.ui.services.search import search_actions
        for query, expected in (("climb",16),("攀爬",16),("jump",61),("aggressive",1),("游泳",106)):
            result=search_actions(query,classes())
            self.assertEqual(result["provider"],"local")
            self.assertIn(expected,result["ids"],query)
    def test_llm_match_validation(self):
        from animalact2.dashboard.ui.services.search import parse_matches
        names={row["class_id"]:row["name"] for row in classes()}
        content='```json\n{"matches":[{"id":16,"reason":"climbing"},{"id":999,"reason":"unknown"},{"name":"Biting","reason":"bite"},{"id":"1"}]}\n```'
        ids,reasons=parse_matches(content,names)
        self.assertEqual(ids,[16,8,1]);self.assertEqual(reasons[16],"climbing")
    def test_llm_success_and_failure(self):
        from unittest.mock import patch
        from animalact2.dashboard.ui.services.search import search_actions
        settings={"api_key":"x","base_url":"https://example.invalid/v1","model":"m"}
        with patch("animalact2.dashboard.ui.services.search._chat",return_value='{"matches":[{"id":118,"reason":"yawn"},{"id":16,"reason":"climb"}]}'):
            result=search_actions("climb or yawn",classes(),settings)
        self.assertEqual((result["provider"],result["ids"],result["model"]),("llm",[118,16],"m"))
        with patch("animalact2.dashboard.ui.services.search._chat",side_effect=RuntimeError("offline")):
            result=search_actions("climb",classes(),settings)
        self.assertEqual(result["provider"],"local");self.assertIn(16,result["ids"]);self.assertIn("offline",result["error"])
    def test_llm_settings(self):
        from unittest.mock import patch
        from animalact2.dashboard.ui.services.search import llm_settings
        settings=llm_settings({"llm":{"api_key":"k","base_url":"https://api.deepseek.com/","model":"deepseek-chat"}})
        self.assertEqual(settings,dict(api_key="k",base_url="https://api.deepseek.com",model="deepseek-chat"))
        with patch.dict(os.environ,{"ANIMALACT2_LLM_API_KEY":"","ANIMALACT2_LLM_BASE_URL":"","ANIMALACT2_LLM_MODEL":""}):
            self.assertIsNone(llm_settings({}))
    def test_search_relevance_sort(self):
        from animalact2.dashboard.ui.services.presentation import score_rows
        rows=[dict(class_id=i,name=f"Class {i}",score=.1*(3-i),selected=False,main_evaluation=True,above_threshold=False,eligible_for_prediction=True) for i in range(3)]
        result={"class_scores":rows}
        out=score_rows(result,order="Search relevance",limit="All 119",query_ids=[2,0],relevance={2:0,0:1})
        self.assertEqual([row["class_id"] for row in out],[2,0])
        self.assertEqual(score_rows(result,query_ids=[]),[])
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
