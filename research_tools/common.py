"""Portable readers retain original identities, masks and content-SHA checks."""
from pathlib import Path
import sys,json,hashlib
import torch
import numpy as np

def bootstrap(workspace):
    root=Path(workspace).resolve()
    if not (root/"animalact2/release_v1/train/manifest.jsonl").is_file():raise FileNotFoundError("Restore the research bundle first")
    for name,module in list(sys.modules.items()):
        if name.startswith("animalact2.") and getattr(module,"__file__",None) and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Restart Python before selecting a different research workspace")
    sys.path.insert(0,str(root));return root

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

class CachedClips(torch.utils.data.Dataset):
    def __init__(self,root,family,split):
        from animalact2.phase1_5_1.dataset import ReleaseClipDataset
        if family not in ("resnet18","videomae") or split not in ("train","valid"):raise ValueError("Only fixed Train/Valid")
        self.root=root;self.family=family;self.split=split;self.records=ReleaseClipDataset(split,release_root=root/"animalact2/release_v1",project_root=root).records
        self.folder=root/"animalact2"/("phase1_5_cache_full" if family=="resnet18" else "phase2_3_full_cache")
        self.context=read(self.folder/"cache/context.json");self.context_sha=digest(self.context)
        plan=read(self.folder/"full_plan.json")
        if plan["context"]!=self.context:raise ValueError("Plan/context mismatch")
        if {r["clip_id"] for r in self.records}!=set(plan["selections"][split]):raise ValueError("Clip set mismatch")
        if sha(root/f"animalact2/release_v1/{split}/manifest.jsonl")!=plan["manifest_SHA256"][split]:raise ValueError("Manifest changed")
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        if self.family=="resnet18":
            from animalact2.phase1_5_cache_pilot.cache import clip_descriptor,read_array
            desc=clip_descriptor(r,self.context)
            folder=self.folder/"cache/clips"/self.split/r["clip_id"]
            x,_=read_array(folder,dict(kind="clip_features",clip_descriptor=desc,clip_key=digest(desc)))
            shape=(r["effective_length"],512)
        else:
            folder=self.folder/"cache"/self.split/r["clip_id"];receipt=read(folder/"receipt.json")
            expected=dict(context_SHA256=self.context_sha,clip_id=r["clip_id"],split=self.split,parent_video_id=r["parent_video_id"],video_path=r["video_path"],
                video_content_SHA256=r["video_sha256"],frame_indices=r["frame_indices"],real_length=r["effective_length"],effective_start=r["effective_start_exact"],effective_end=r["effective_end_exact"],
                label_and_provenance_SHA256=digest(dict(target=r["target_119"],evidence=r["label_evidence"])))
            if receipt["descriptor"]!=expected or receipt["key"]!=digest(expected) or receipt["target_119"]!=r["target_119"]:raise ValueError("Label/frame provenance mismatch")
            if receipt["features_SHA256"]!=sha(folder/"features.npy"):raise ValueError("Feature content changed")
            x=np.load(folder/"features.npy",allow_pickle=False);shape=((r["effective_length"]+1)//2,768)
        if x.dtype!=np.float32 or x.shape!=shape or not np.isfinite(x).all():raise ValueError("Invalid real-length FP32 features")
        return dict(frames=torch.from_numpy(x.copy()),target=torch.tensor(r["target_119"],dtype=torch.float32),clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split)

def collate(samples):
    lengths=torch.tensor([len(s["frames"]) for s in samples]);x=torch.zeros(len(samples),int(lengths.max()),samples[0]["frames"].shape[1]);mask=torch.arange(x.shape[1])[None]<lengths[:,None]
    for i,s in enumerate(samples):x[i,:lengths[i]]=s["frames"]
    return dict(frames=x,lengths=lengths,valid_frame_mask=mask,padding_mask=~mask,target=torch.stack([s["target"] for s in samples]),
        clip_ids=[s["clip_id"] for s in samples],parent_video_ids=[s["parent_video_id"] for s in samples],splits=[s["split"] for s in samples])

class FeatureView:
    def __init__(self,model):self.model=model
    def __call__(self,x,lengths,mask):
        return self.model.forward_features(x,lengths,mask) if hasattr(self.model,"forward_features") else self.model(x,lengths,mask)
    def train(self,mode=True):self.model.train(mode);return self
    def eval(self):self.model.eval();return self
    def parameters(self):return self.model.parameters()
    def named_parameters(self):return self.model.named_parameters()

def load_encoder(root,device="cpu"):
    # Resolve only the runtime pointer, never edit sealed source/context or repeat frames.
    source=read(root/".cache/videomae_source.json");source["path"]=str(root/read(root/"research_locations.json")["videomae_directory"])
    folder=root/".runtime";folder.mkdir(exist_ok=True);pointer=folder/"portable_videomae_source.json";pointer.write_text(json.dumps(source),encoding="utf-8")
    import animalact2.phase2_2.adapter as adapter
    adapter.SOURCE=pointer
    return adapter.FrozenVideoMAEAdapter(device)
