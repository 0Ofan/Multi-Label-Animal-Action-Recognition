"""Loopback JSON HTTP API; raw video upload, bounded body, no model downloads."""
import argparse
import json
import tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs,unquote
from .registry import model_info,OUT
from .video import MAX_BYTES,SUPPORTED,inspect_video
from .evaluation_data import evaluation
from .schema import validate_prediction

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address):
        super().__init__(address,Handler)
        from .predictor import Runtime
        self.runtime=Runtime()

class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup(); self.connection.settimeout(300)
    def send_json(self,code,value):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p=urlparse(self.path); q=parse_qs(p.query)
        try:
            if p.path=="/health": value=dict(status="ok",version="AnimalAct 2.0",loaded_devices=list(self.server.runtime.predictors))
            elif p.path=="/api/models": value=[model_info(k) for k in ("attention","bigru_mean")]
            elif p.path=="/api/evaluation": value=evaluation(q.get("kind",["summary"])[0],q.get("model",["model3"])[0],int(q.get("seed",[42])[0]))
            else: self.send_json(404,{"error":"Endpoint not found"}); return
            self.send_json(200,value)
        except (ValueError,KeyError) as exc: self.send_json(400,{"error":str(exc)})
        except Exception as exc: self.send_json(500,{"error":"Experiment/model information could not be loaded: "+str(exc)})
    def do_POST(self):
        p=urlparse(self.path); q=parse_qs(p.query)
        if p.path!="/api/predict": self.send_json(404,{"error":"Endpoint not found"}); return
        try:
            key=q.get("model",["attention"])[0]; model_info(key)
            device=q.get("device",["cuda"])[0]
            if device not in ("cuda","cpu"): raise ValueError("Device must be cuda or cpu")
            n=int(self.headers.get("Content-Length","0"))
            if not 0<n<=MAX_BYTES:
                self.close_connection=True; self.send_json(413,{"error":"Upload must contain 1 byte to 200 MiB"}); return
            name=Path(unquote(self.headers.get("X-Filename","upload.mp4")).replace("\\","/")).name
            if Path(name).suffix.lower() not in SUPPORTED: raise ValueError("Unsupported video format")
            start=float(q.get("start",[0])[0]); end=q.get("end",[None])[0]; end=None if end is None else float(end)
            temp=OUT/".runtime/uploads"; temp.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="request_",dir=temp) as directory:
                path=Path(directory)/("upload"+Path(name).suffix.lower())
                data=self.rfile.read(n)
                if len(data)!=n: raise ValueError("Incomplete video upload")
                path.write_bytes(data)
                inspect_video(path)
                result=self.server.runtime.predict(path,key,device,start,end)
                result["video_id"]=Path(name).stem
                self.send_json(200,validate_prediction(result))
        except (ValueError,KeyError) as exc:
            self.close_connection=True; self.send_json(400,{"error":str(exc)})
        except Exception as exc:
            self.close_connection=True; self.send_json(500,{"error":"Model loading or inference failed: "+str(exc)})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--port",type=int,default=8765); args=parser.parse_args()
    server=Server(("127.0.0.1",args.port)); print(f"AnimalAct 2.0 API http://127.0.0.1:{args.port}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
if __name__=="__main__": main()
