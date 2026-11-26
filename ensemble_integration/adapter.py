import json,time
from pathlib import Path
import numpy as np
from .backend import Engine

class IntegratedEngine:
    def __init__(self):
        self.engine=Engine()
    def predict(self,path):
        from animalact2.dashboard.registry import classes
        start=time.perf_counter()
        value=self.engine.predict(path)
        threshold=float(np.float32(value['threshold']))
        rows=[]
        for original,item in zip(classes(),value['rows']):
            if original['name']!=item['action']:raise ValueError('Fusion vocabulary differs.')
            above=item['score']>=threshold
            rows.append(dict(original,score=item['score'],above_threshold=above,selected=above,
                             eligible_for_prediction=True,learning_status='Whole_video_BCE',
                             bigru_mean_score=item['bigru_mean'],attention_score=item['attention']))
        return dict(schema_version=2,video_id=Path(path).stem,model_id='resnet_ensemble',
            model_name='ResNet-18 + weighted BiGRU ensemble',device='cpu',evaluation_unit='whole_video',
            threshold=threshold,class_scores=rows,
            logits=[float(np.log(np.clip(r['score'],1e-8,1-1e-8)/(1-np.clip(r['score'],1e-8,1-1e-8)))) for r in rows],
            logits_semantics='Logit of fused score; not a separate learned network output.',
            predicted_labels=[r['name'] for r in sorted(rows,key=lambda r:(-r['score'],r['class_id'])) if r['selected']],
            unsupported_class_ids=[],calibrated=False,feature_shape=[1,16,512],
            inference_time=time.perf_counter()-start,fusion_weights=value['weights'])
