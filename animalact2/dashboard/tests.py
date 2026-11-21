"""No training / Test reads. Run after starting both local services."""
import io,json,math,unittest,csv
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import torch
from .registry import ROOT,OUT,model_info,classes
from .video import sample_indices,prepare_video
from .schema import validate_prediction
from .evaluation_data import evaluation
from .ui.services.state import accept_upload,clear
from .ui.services.presentation import export_csv,score_rows

class Checks(unittest.TestCase):
    def test_registry(self):
        self.assertEqual((model_info()["best_epoch"],model_info()["threshold"]),(21,.12))
        self.assertEqual((model_info("bigru_mean")["best_epoch"],model_info("bigru_mean")["threshold"]),(41,.07))
    def test_mapping(self):
        self.assertEqual(len(classes()),119); self.assertEqual(sum(r["main_evaluation"] for r in classes()),31)
    def test_real_lengths(self):
        for n in range(1,101):
            ids,_=sample_indices(n,30.)
            self.assertEqual(len(ids),min(n,16)); self.assertEqual(ids,sorted(set(ids)))
    def test_half_open(self):
        ids,_=sample_indices(100,10.,.3,.6); self.assertEqual(ids,[3,4,5])
    def test_bad_windows(self):
        for a,b in [(-1,1),(1,1),(2,1),(0,11),(math.nan,1),(.01,.02)]:
            with self.assertRaises(ValueError): sample_indices(100,10.,a,b)
    def test_heads_and_mask(self):
        from animalact2.phase2_2.heads import VideoMAEProposalHead
        for key in ("mean","bigru_mean","bigru_attention"):
            h=VideoMAEProposalHead(key).eval()
            x=torch.randn(3,8,768); lens=torch.tensor([1,3,8]); mask=torch.arange(8)[None]<lens[:,None]
            y=h(x,lens,mask); x[~mask]=1e4
            torch.testing.assert_close(y,h(x,lens,mask)); self.assertEqual(tuple(y.shape),(3,119))
    def test_odd_geometry(self):
        from animalact2.phase2_2.adapter import align_pixels
        for n in (1,3,15,16):
            x=torch.randn(1,n,3,224,398); a,g=align_pixels(x)
            self.assertEqual(g["grid"],[(n+1)//2,14,25]); torch.testing.assert_close(a[:,:,:n,:,:398],x.permute(0,2,1,3,4))
    def test_evaluation_counts(self):
        self.assertEqual(len(evaluation("summary")["rows"]),3)
        self.assertEqual(len(evaluation("runs")["rows"]),9)
        self.assertEqual(len(evaluation("per_class")["rows"]),558)
    def test_all_confusions(self):
        for m in ("model1","model2","model3"):
            for s in (42,123,2026):
                rows=evaluation("confusion",m,s)["rows"];self.assertEqual(len(rows),31)
                for row in rows:self.assertEqual(sum(int(row[k]) for k in ("TP","FP","TN","FN")),979)
    def test_no_test_access(self):
        with self.assertRaises(ValueError):evaluation("Test")
        with self.assertRaises(ValueError):evaluation("history","../../test",42)
        with self.assertRaises(ValueError):evaluation("history","model3",99)
    def test_upload_rejects_corrupt(self):
        state={};accept_upload(state,"broken.mp4",b"bad")
        self.assertIn("recognition_error",state);self.assertNotIn("recognition",state)
        clear(state);self.assertNotIn("upload",state)
    def test_state_and_exports(self):
        result=json.loads((OUT/"acceptance/api_attention.json").read_text())
        validate_prediction(result)
        rows=list(csv.DictReader(io.StringIO(export_csv({"filename":"upload.mp4","result":result}).decode("utf-8-sig"))))
        self.assertEqual(len(rows),119);self.assertIn("eligible_for_prediction",rows[0])
        self.assertEqual(len(score_rows(result,scope="Main 31 classes")),31)
        self.assertFalse(result["class_scores"][23]["selected"])
    def test_http_errors(self):
        tests=[("/api/predict?model=wrong",b"bad",{},400),("/api/predict",b"",{},413),
               ("/api/predict",b"bad",{"X-Filename":"x.exe"},400),("/api/predict",b"bad",{"X-Filename":"x.mp4"},400),
               ("/api/predict?device=invalid",b"bad",{},400)]
        for path,data,headers,status in tests:
            with self.assertRaises(HTTPError) as caught:urlopen(Request("http://127.0.0.1:8765"+path,data=data,headers=headers),timeout=30)
            self.assertEqual(caught.exception.code,status)
    def test_model_load_failure(self):
        from unittest.mock import patch
        from .predictor import Predictor
        instance=Predictor.__new__(Predictor);instance.heads={}
        with patch("animalact2.dashboard.predictor.sha",return_value="wrong"):
            with self.assertRaisesRegex(ValueError,"Checkpoint SHA256 mismatch"):
                instance.load_head("attention")
    def test_oversized_upload(self):
        import http.client
        client=http.client.HTTPConnection("127.0.0.1",8765,timeout=30)
        client.request("POST","/api/predict",body=b"",headers={"Content-Length":str(200*1024**2+1)})
        self.assertEqual(client.getresponse().status,413);client.close()
    def test_pages(self):
        from streamlit.testing.v1 import AppTest
        for page in ("video","results","evaluation"):
            app=AppTest.from_file(str(OUT/"ui/app.py"),default_timeout=60)
            app.query_params["page"]=page;app.run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertFalse(any("could not be loaded" in e.value for e in app.error))

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    text=stream.getvalue();(OUT/"acceptance/automated_tests.txt").write_text(text,encoding="utf-8");print(text)
    (OUT/"acceptance/automated_tests.json").write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),indent=2))
    raise SystemExit(not result.wasSuccessful())
