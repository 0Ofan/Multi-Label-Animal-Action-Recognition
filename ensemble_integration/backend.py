"""Portable CPU inference using unchanged original classification heads."""
import hashlib
import json
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights
from .core_models.model1 import Model1
from .core_models.model2 import Model2
from .core_models.model3 import Model3

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Engine:
    def __init__(self):
        from deployment.final_artifacts import ensure_final_runtime
        ensure_final_runtime()
        torch.set_num_threads(4)
        self.lock=threading.Lock()
        self.policy=json.loads((ROOT/'experiment/policy.json').read_text(encoding='utf-8'))
        path=ROOT/'weights/resnet18-f37072fd.pth'
        if sha(path)!='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
            raise ValueError('ResNet-18 权重校验失败，请重新解压交付包。')
        self.encoder=resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
        self.encoder.fc=nn.Identity()
        self.encoder.eval().requires_grad_(False)
        self.preprocess=ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.heads={}
        for name,weight in zip(self.policy['models'],self.policy['weights']):
            if not weight:continue
            path=ROOT/'weights'/(name+'.pt')
            if sha(path)!=self.policy['checkpoint_sha256'][name]:raise ValueError('分类头校验失败：'+name)
            cp=torch.load(path,weights_only=True,map_location='cpu')
            if cp['classes']!=self.policy['classes'] or cp['manifest_sha256']!=self.policy['manifest_sha256']:
                raise ValueError('模型来源或词表不一致。')
            head={'mean':Model1,'bigru_mean':Model2,'bigru_attention':Model3}[name](backbone=nn.Identity())
            head.load_state_dict(cp['state_dict'])
            self.heads[name]=head.eval().requires_grad_(False)

    def predict(self,path):
        with self.lock,torch.inference_mode():
            cap=cv2.VideoCapture(str(path))
            try:
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                if total<1 or fps<=0:raise ValueError('无法读取视频，请尝试 MP4 格式。')
                indices=((np.arange(16)+.5)*total/16).astype(int)
                targets=set(indices.tolist());frames={}
                for i in range(int(indices[-1])+1):
                    if not cap.grab():break
                    if i in targets:
                        ok,frame=cap.retrieve()
                        if ok:frames[i]=self.preprocess(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                if len(frames)!=len(targets):raise ValueError('视频解码不完整。')
            finally:cap.release()
            features=self.encoder(torch.stack([frames[i] for i in indices]))[None]
            parts={name:head.forward_features(features,torch.tensor([16]),torch.ones((1,16),dtype=torch.bool))[0].sigmoid().numpy()
                   for name,head in self.heads.items()}
            weights=dict(zip(self.policy['models'],self.policy['weights']))
            scores=sum(weights[name]*value.astype('float64') for name,value in parts.items())
            eligible=set(self.policy['eligible_class_ids']);threshold=self.policy['threshold']
            rows=[dict(action=c,score=float(scores[i]),bigru_mean=float(parts['bigru_mean'][i]),
                  attention=float(parts['bigru_attention'][i]),selected=i in eligible and bool(scores[i]>=threshold),eligible=i in eligible)
                  for i,c in enumerate(self.policy['classes'])]
            return dict(predicted_actions=[r['action'] for r in sorted(rows,key=lambda r:-r['score']) if r['selected']],
                rows=rows,duration=total/fps,threshold=threshold,weights=weights,
                note='整视频多标签动作分类，分数未校准，不输出动作时间区间。')
