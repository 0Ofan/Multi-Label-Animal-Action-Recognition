from __future__ import annotations
import ctypes
from ctypes import wintypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_json(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError("Pilot-only output")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def forbidden(path):
    text="/"+path.as_posix().lower()
    return any(s in text for s in ("/videos/test/","/annotations/test/","/test_metadata_only/","predictions","model6"))


def guard(event,args):
    if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network in cache pilot")
    if event!="open" or not isinstance(args[0],(str,bytes)):return
    path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
    write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
    if write and not path.is_relative_to(OUT):raise RuntimeError("Write outside cache pilot: "+str(path))
    if path.is_relative_to(ROOT) and not path.is_relative_to(OUT):
        if forbidden(path):raise RuntimeError("Out-of-scope Test/prediction/Model6 read")
        if path.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors") and path!=ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth":
            raise RuntimeError("Old trained weights or feature caches cannot be loaded")


def enable_guard():sys.addaudithook(guard)


def snapshot():
    rr={};skip={".venv",".git",".codex",".agents",".aws","node_modules","__pycache__"}
    for directory,sub,files in os.walk(ROOT):
        p=Path(directory);sub[:]=[n for n in sub if n not in skip and "model6" not in n.lower() and (p/n).resolve()!=OUT]
        for name in files:
            path=p/name
            if "model6" in name.lower() or path.is_symlink():continue
            st=path.stat();rr[path.relative_to(ROOT).as_posix()]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return rr


def historical_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for directory in (ROOT/"animalact2").iterdir() if directory.is_dir() and directory.resolve()!=OUT
            for p in directory.rglob("*") if p.is_file() and not forbidden(p) and p.suffix.lower() not in (".pt",".pth",".npy",".npz",".safetensors")}


def capture():
    path=OUT/"protection_baseline.json"
    if path.exists():raise FileExistsError("Keep the original pilot baseline")
    obj=dict(metadata=snapshot(),historical_SHA256=historical_hashes(),official_weights_SHA256=sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth"),started_unix=time.time())
    write_json(path,obj);print(json.dumps(dict(metadata_files=len(obj["metadata"]),hashed_non_Test_non_tensor_files=len(obj["historical_SHA256"]))))


def verify():
    obj=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot()==obj["metadata"] and historical_hashes()==obj["historical_SHA256"]
    assert sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth")==obj["official_weights_SHA256"]
    result=dict(status="passed",historical_metadata_files=len(obj["metadata"]),hashed_non_Test_non_tensor_files=len(obj["historical_SHA256"]),release_models_and_formal_configs_unchanged=True,Test_content_read=False)
    write_json(OUT/"protection_verification.json",result);return result


class Counters(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]


def process_rss(pid):
    kernel=ctypes.WinDLL("kernel32",use_last_error=True);psapi=ctypes.WinDLL("psapi",use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    handle=kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle:return 0
    try:
        data=Counters();data.cb=ctypes.sizeof(data)
        return int(data.WorkingSetSize) if psapi.GetProcessMemoryInfo(handle,ctypes.byref(data),data.cb) else 0
    finally:kernel.CloseHandle(handle)


class RAMMonitor:
    """Sample parent+own multiprocessing workers RSS; shared pages may count twice."""
    def __enter__(self):
        self.peak_parent=0;self.peak_sum=0;self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                parent=process_rss(os.getpid());children=sum(process_rss(p.pid) for p in multiprocessing.active_children() if p.pid)
                self.peak_parent=max(self.peak_parent,parent);self.peak_sum=max(self.peak_sum,parent+children)
                self.stop.wait(.02)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self

    def __exit__(self,*args):self.stop.set();self.thread.join()

    def result(self):return dict(parent_RSS_peak_bytes=self.peak_parent,parent_plus_workers_RSS_sum_peak_bytes=self.peak_sum,interval_ms=20,scope="Working-set/RSS; shared pages counted per process, not unique physical RAM")


if __name__=="__main__":
    enable_guard()
    if sys.argv[1:]==["capture"]:capture()
    elif sys.argv[1:]==["verify"]:print(json.dumps(verify(),indent=2))
