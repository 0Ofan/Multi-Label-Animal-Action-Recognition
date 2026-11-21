"""Verify course Train/Valid source contents; ignore timestamps, never open Test."""
import argparse,json
from pathlib import Path
from common import bootstrap,sha
p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);a=p.parse_args();root=bootstrap(a.workspace)
seen={}
for split in ("train","valid"):
    path=root/f"animalact2/release_v1/{split}/manifest.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        for kind in ("video","annotation"):
            rel=r[kind+"_path"];actual=(a.data_root/Path(rel).relative_to("AnimalAct")).resolve()
            expected=r[kind+"_sha256"]
            if rel in seen:
                if seen[rel]!=expected:raise ValueError("Conflicting source identity")
                continue
            if sha(actual)!=expected:raise ValueError("Different source file: "+rel)
            seen[rel]=expected
print(json.dumps(dict(status="passed",source_files_checked=len(seen),Train_Valid_only=True,Test_read=False)))
