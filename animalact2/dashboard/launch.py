"""Start/stop only this Dashboard's local services; hide Windows helper windows."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import ctypes
from ctypes import wintypes
from .registry import OUT,ROOT

PIDFILE=OUT/".runtime/servers.json"
def start():
    if PIDFILE.exists() and any(alive(r) for r in json.loads(PIDFILE.read_text())):
        raise RuntimeError("An owned Dashboard process is still running. Use status/stop first")
    PIDFILE.parent.mkdir(exist_ok=True,parents=True)
    rows=[]
    for name,args in [("backend",["-m","animalact2.dashboard.api","--port","8765"]),("frontend",["-m","streamlit","run","animalact2/dashboard/ui/app.py","--server.port","8502","--server.address","127.0.0.1"])]:
        with (PIDFILE.parent/(name+".log")).open("w",encoding="utf-8") as f:
            p=subprocess.Popen([sys.executable,"-u","-B","-X","utf8",*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        rows.append(dict(name=name,pid=p.pid,created=stamp(p.pid)))
    PIDFILE.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print("Frontend http://127.0.0.1:8502 | API http://127.0.0.1:8765",flush=True)
    print(json.dumps(rows),flush=True)
def stamp(pid):
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    h=kernel.OpenProcess(0x1000,False,pid)
    if not h:return None
    try:
        values=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(h,*[ctypes.byref(x) for x in values]):return None
        return values[0].dwHighDateTime*2**32+values[0].dwLowDateTime
    finally:kernel.CloseHandle(h)
def alive(row):
    return row.get("created") is not None and stamp(row["pid"])==row["created"]
def terminate(pid):
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    k.OpenProcess.restype=wintypes.HANDLE
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(1,False,pid)
    if not h:raise OSError(ctypes.get_last_error())
    try:k.TerminateProcess(h,0)
    finally:k.CloseHandle(h)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["start","stop","status"]); args=parser.parse_args()
    if args.action=="start":start();return
    rows=json.loads(PIDFILE.read_text()) if PIDFILE.exists() else []
    for row in rows:
        running=alive(row)
        if args.action=="stop" and running: terminate(row["pid"])
        print(row["name"],row["pid"],"stopped" if args.action=="stop" else "running" if running else "not running")
if __name__=="__main__":main()
