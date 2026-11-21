import csv,json,gc,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from animalact2.phase1_5.losses import UnsupportedClassPolicy
from animalact2.phase1_5.evaluation import evaluate_outputs,select_validation_threshold
from .support import OUT,ROOT,enable_guard,write_json,sha
from .data import CachedFeatureDataset,collate_features
from .models import build_model,expected_config,KINDS
from . import checkpoint
from animalact2.phase1_5.config import config_hash

def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    enable_guard();torch.set_num_threads(4);start=time.perf_counter()
    if (OUT/'analysis_complete.json').exists():raise FileExistsError('Keep completed analysis')
    datasets={s:CachedFeatureDataset(s) for s in ('train','valid')}
    counts=torch.tensor([r['target_119'] for r in datasets['train'].records]).sum(0)
    rows=[];aprows=[];groups=[];warnings=[]
    os.environ['MPLCONFIGDIR']=str(OUT/'.runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for seed in (42,123,2026):
      for name in KINDS:
        key=f'{name}_seed{seed}'
        d=ROOT/f'animalact2/phase2_4/runs/{name}_bce_seed42' if seed==42 else OUT/'runs'/key
        result=json.loads((d/'results.json').read_text());assert result['status']=='completed'
        c=expected_config(name,seed);assert result['config']==c
        state=torch.load(d/'best.pt',map_location='cpu',weights_only=True)
        assert state['config']==c and state['epoch']==result['best_epoch'] and not state['diagnostic']
        assert state['config_SHA256']==config_hash(c) and sha(d/'best.pt')==result['checkpoint_SHA256']
        if seed!=42:assert state['first_round_implementation']==checkpoint.fingerprints()
        assert state['cache_provenance']==result['cache_provenance']
        model=build_model(c).cuda().eval();model.load_state_dict(state['head_state'],strict=True)
        policy=UnsupportedClassPolicy(model,counts);policy.load_state_dict(state['class_policy'])
        for n,v in policy.anchors.items():assert torch.equal(dict(model.named_parameters())[n].detach().cpu()[[23]],v)
        hist=json.loads((d/'history.json').read_text());assert len(hist)==result['completed_epochs']
        assert state['epoch']==max(hist,key=lambda r:r['valid_mAP'])['epoch']
        metrics={}
        for split,ds in datasets.items():
          output=OUT/'fixed_checkpoint'/key/f'{split}.pt'
          if output.exists():
            pred=torch.load(output,map_location='cpu',weights_only=True);assert pred['checkpoint_SHA256']==sha(d/'best.pt')
          else:
            zs=[];ys=[];ids=[]
            with torch.inference_mode():
              for b in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate_features):
                zs.append(model(b['frames'].cuda(),b['lengths'],b['valid_frame_mask'].cuda()).cpu());ys.append(b['target']);ids+=b['clip_ids']
            pred=dict(logits=torch.cat(zs),target=torch.cat(ys),clip_ids=ids,checkpoint_SHA256=sha(d/'best.pt'))
            output.parent.mkdir(parents=True,exist_ok=True);torch.save(pred,output)
          z,y=pred['logits'],pred['target'];assert pred['clip_ids']==[r['clip_id'] for r in ds.records]
          assert torch.equal(y,torch.tensor([r['target_119'] for r in ds.records],dtype=torch.float32))
          threshold=result['valid']['threshold']
          m=evaluate_outputs(z,y,[r['parent_video_id'] for r in ds.records],policy,threshold=threshold);metrics[split]=m
          write_json(output.with_suffix('.json'),m)
          if split=='valid':
            assert abs(m['main_mAP']-result['valid']['main_mAP'])<1e-12
            assert select_validation_threshold(z,y,split='valid')['threshold']==threshold
          for r in m['per_class']:
            if r['main_evaluation']:aprows.append(dict(model=name,seed=seed,split=split,class_id=r['class_id'],name=r['name'],positives=r['positive_clips'],AP=r['AP'],Precision=r['Precision'],Recall=r['Recall'],F1=r['F1']))
          for group,selector in [('length1_3',lambda r:r['effective_length']<=3),('length4_15',lambda r:4<=r['effective_length']<=15),('length16',lambda r:r['effective_length']==16),('coverage_warning',lambda r:bool(r['sampled_label_coverage_warnings'])),('no_coverage_warning',lambda r:not r['sampled_label_coverage_warnings'])]:
            ix=[i for i,r in enumerate(ds.records) if selector(r)]
            gm=evaluate_outputs(z[ix],y[ix],[ds.records[i]['parent_video_id'] for i in ix],policy,threshold=threshold,require_all_main_classes=False)
            eligible=[r for r in gm['per_class'] if r['main_evaluation'] and r['AP'] is not None]
            groups.append(dict(model=name,seed=seed,split=split,group=group,N=len(ix),full31_mAP=gm['main_mAP'],supported_main_mAP=float(np.mean([r['AP'] for r in eligible])) if eligible else None,supported_main_classes=len(eligible),missing_main_ids=json.dumps(gm['missing_main_class_ids']),Macro_F1_fixed31=gm['main_macro_f1'],BCE118=gm['BCE_Train_supported118'],BCE119=gm['BCE119']))
          pairs=[(i,k) for i,r in enumerate(ds.records) for k in r['sampled_label_coverage_warnings'] if k in m['main_class_ids']]
          scores=z.double().sigmoid()
          warnings.append(dict(model=name,seed=seed,split=split,warning_clips=sum(bool(r['sampled_label_coverage_warnings']) for r in ds.records),main31_warning_positive_pairs=len(pairs),main31_missed_sampling_label_recall=sum(float(scores[i,k])>=threshold for i,k in pairs)/len(pairs),mean_warning_label_probability=np.mean([float(scores[i,k]) for i,k in pairs])))
        tm,vm=metrics['train'],metrics['valid'];lossmin=min(hist,key=lambda r:r['valid_objective'])
        rows.append(dict(model=name,head=KINDS[name],seed=seed,best_epoch=state['epoch'],epochs=len(hist),threshold=result['valid']['threshold'],Train_mAP=tm['main_mAP'],Valid_mAP=vm['main_mAP'],mAP_gap=tm['main_mAP']-vm['main_mAP'],Train_Macro_F1=tm['main_macro_f1'],Valid_Macro_F1=vm['main_macro_f1'],F1_gap=tm['main_macro_f1']-vm['main_macro_f1'],Train_BCE118=tm['BCE_Train_supported118'],Valid_BCE118=vm['BCE_Train_supported118'],Train_BCE119=tm['BCE119'],Valid_BCE119=vm['BCE119'],BCE118_gap=vm['BCE_Train_supported118']-tm['BCE_Train_supported118'],min_Valid_BCE_epoch=lossmin['epoch'],min_Valid_BCE118=lossmin['valid_objective'],best_mAP_minus_min_BCE_epoch=state['epoch']-lossmin['epoch'],training_seconds=result['resources']['wall_seconds'],GPU_peak_MiB=result['resources']['GPU_allocated_peak_bytes']/2**20,RAM_parent_MiB=result['resources']['CPU_RSS_parent_peak_bytes']/2**20,RAM_workers_sum_MiB=result['resources']['CPU_RSS_parent_workers_sum_peak_bytes']/2**20,seed42_reused=seed==42))
        if seed!=42:
          fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
          for ax,(a,b,title) in zip(axes,[('train_online_objective','valid_objective','BCE118: Train online / Valid fixed epoch'),('train_online_mAP','valid_mAP','31-class mAP: Train online / Valid fixed epoch')]):
            ax.plot([r['epoch'] for r in hist],[r[a] for r in hist],label='Train online');ax.plot([r['epoch'] for r in hist],[r[b] for r in hist],label='Valid');ax.axvline(state['epoch'],ls='--',c='gray');ax.set(title=title,xlabel='Epoch');ax.legend();ax.grid(alpha=.2)
          fig.suptitle(key);fig.savefig(d/'training_curves.png',dpi=140);fig.savefig(d/'training_curves_diagnostic.svg');plt.close(fig)
        print(json.dumps(dict(fixed_checkpoint_evaluated=key,Train_mAP=tm['main_mAP'],Valid_mAP=vm['main_mAP'])),flush=True)
        del model,state,pred,z,y;gc.collect();torch.cuda.empty_cache()
    for row in rows:
      d=ROOT/f"animalact2/phase2_4/runs/{row['model']}_bce_seed42" if row['seed']==42 else OUT/'runs'/f"{row['model']}_seed{row['seed']}"
      hh=json.loads((d/'history.json').read_text());bb=min(hh,key=lambda r:r['valid_BCE119'])
      row['min_Valid_BCE119_epoch']=bb['epoch'];row['min_Valid_BCE119']=bb['valid_BCE119'];row['best_mAP_minus_min_BCE119_epoch']=row['best_epoch']-bb['epoch']
    csvout('all_seed_results.csv',rows);csvout('per_class_AP.csv',aprows);csvout('length_and_coverage_groups.csv',groups);csvout('coverage_warning_labels.csv',warnings)
    summary=[]
    for name in KINDS:
      subset=[r for r in rows if r['model']==name];row=dict(model=name,head=KINDS[name],seeds='42,123,2026',N=3,std_ddof=1)
      for k in ('Train_mAP','Valid_mAP','Train_Macro_F1','Valid_Macro_F1','Train_BCE118','Valid_BCE118','Train_BCE119','Valid_BCE119','mAP_gap','F1_gap','BCE118_gap'):
        row[k+'_mean']=float(np.mean([r[k] for r in subset]));row[k+'_std']=float(np.std([r[k] for r in subset],ddof=1))
      summary.append(row)
    csvout('summary_mean_std.csv',summary)
    stability=[]
    for name in KINDS:
      for split in ('train','valid'):
       for cid in sorted({r['class_id'] for r in aprows}):
        a=[r for r in aprows if r['model']==name and r['split']==split and r['class_id']==cid]
        stability.append(dict(model=name,split=split,class_id=cid,name=a[0]['name'],positives=a[0]['positives'],AP_mean=float(np.mean([r['AP'] for r in a])),AP_std=float(np.std([r['AP'] for r in a],ddof=1))))
    csvout('per_class_AP_stability.csv',stability)
    comparison=[]
    for r in summary:
      old=json.loads((ROOT/f"animalact2/first_round/runs/{r['model']}_bce_seed42/results.json").read_text())['valid']
      comparison.append(dict(model=r['model'],ResNet_seed42_Valid_mAP=old['main_mAP'],VideoMAE_Valid_mAP_mean=r['Valid_mAP_mean'],VideoMAE_Valid_mAP_std=r['Valid_mAP_std'],mAP_difference=r['Valid_mAP_mean']-old['main_mAP'],ResNet_seed42_Valid_F1=old['main_macro_f1'],VideoMAE_Valid_F1_mean=r['Valid_Macro_F1_mean'],VideoMAE_Valid_F1_std=r['Valid_Macro_F1_std']))
    for r in comparison:
      old=json.loads((ROOT/f"animalact2/first_round/runs/{r['model']}_bce_seed42/results.json").read_text())['valid']
      sm=next(x for x in summary if x['model']==r['model'])
      for k in ('BCE118','BCE119'):
        r['ResNet_seed42_'+k]=old['BCE_Train_supported118' if k=='BCE118' else 'BCE119']
        r['VideoMAE_'+k+'_mean']=sm['Valid_'+k+'_mean'];r['VideoMAE_'+k+'_std']=sm['Valid_'+k+'_std']
    csvout('ResNet_comparison.csv',comparison)
    paired=[]
    for seed in (42,123,2026):
      for a,b in (('model2','model1'),('model3','model1'),('model3','model2')):
        ra=next(r for r in rows if r['model']==a and r['seed']==seed);rb=next(r for r in rows if r['model']==b and r['seed']==seed)
        aa={r['class_id']:r['AP'] for r in aprows if r['model']==a and r['seed']==seed and r['split']=='valid'}
        bb={r['class_id']:r['AP'] for r in aprows if r['model']==b and r['seed']==seed and r['split']=='valid'}
        paired.append(dict(seed=seed,model_a=a,model_b=b,Valid_mAP_delta=ra['Valid_mAP']-rb['Valid_mAP'],Valid_F1_delta=ra['Valid_Macro_F1']-rb['Valid_Macro_F1'],classes_AP_up=sum(aa[k]>bb[k] for k in aa),classes_AP_down=sum(aa[k]<bb[k] for k in aa),classes_AP_equal=sum(aa[k]==bb[k] for k in aa)))
    csvout('head_paired_changes.csv',paired)

    table='\n'.join(f"| {r['head']} | {100*r['Valid_mAP_mean']:.2f} ± {100*r['Valid_mAP_std']:.2f}% | {100*r['Valid_Macro_F1_mean']:.2f} ± {100*r['Valid_Macro_F1_std']:.2f}% | {r['Valid_BCE118_mean']:.6f} ± {r['Valid_BCE118_std']:.6f} | {100*r['mAP_gap_mean']:.2f} |" for r in summary)
    detail='\n'.join(f"| {r['head']} | {r['seed']} | {r['best_epoch']}/{r['epochs']} | {100*r['Train_mAP']:.2f}% | {100*r['Valid_mAP']:.2f}% | {100*r['Train_Macro_F1']:.2f}% | {100*r['Valid_Macro_F1']:.2f}% | {r['Train_BCE118']:.5f}/{r['Valid_BCE118']:.5f} | {r['min_Valid_BCE_epoch']} |" for r in rows)
    best=max(summary,key=lambda r:r['Valid_mAP_mean'])
    text=f"""# Phase 2.5 多种子验证与过拟合分析

已复用Phase2.4 Seed42三组结果，仅补充Seed123、2026六组。结构、缓存、BCE118、AdamW、LR1e-3、WD1e-4、Batch8×累积4、2轮Warmup+Cosine至10%、最多50轮、Patience10、Valid31类mAP选模与Valid阈值规则均不变。无增强、微调、特征提取或Test读取。

## 三种子汇总

均值±样本标准差（ddof=1，N=3），指标为最佳Checkpoint固定权重评价。

| 头 | Valid mAP | Valid Macro-F1 | Valid BCE118 | Train−Valid mAP差/百分点 |
|---|---|---|---|---|
{table}

按Valid mAP均值，首选{best['head']}。三种子只反映本次重复性，不证明统计显著性或Test性能。

## 每种子固定Checkpoint结果

两Split使用同一个最佳权重和该权重的Valid所选阈值；Train不重新选阈值。

| 头 | Seed | 最佳/完成轮 | Train mAP | Valid mAP | Train F1 | Valid F1 | Train/Valid BCE118 | 最低Valid BCE轮 |
|---|---|---|---|---|---|---|---|---|
{detail}

完整BCE119、阈值、资源、耗时及差距见all_seed_results.csv。训练Online指标是Epoch内混合参数，不能冒充这里的固定Checkpoint Train指标。六组history.csv及training_curves.png/svg独立保留；固定Train/Valid预测及指标在fixed_checkpoint/，Seed42原文件不修改。

## 过拟合判读

固定Checkpoint的Train/Valid差距比Online曲线更适合判断泛化。结合表中差距、Train Loss下降与Valid Loss回升，以及最佳mAP轮晚于最低Valid BCE轮，说明继续优化排序时概率拟合可能已恶化。mAP和BCE目标不同，本轮按原协议选模，不事后改选较早Checkpoint。

31类逐类AP和跨种子标准差见per_class_AP.csv与per_class_AP_stability.csv；支持数随种子相同，较大AP标准差应结合少量正例解读，不能仅按单次AP提升选配置。

长度诊断分为真实1–3、4–15和16帧，而非tubelet长度。见length_and_coverage_groups.csv：若组内缺少主类正例，full31_mAP为空；supported_main_mAP仅为该组有正例类别的均值，不能与完整31类mAP直接比较或用于选模。表中同时保存支持类别数与缺失ID，Macro-F1仍按固定31类定义。

标签覆盖警告标签本身在未命中帧中有密集支持，但输入采样未命中。coverage_warning_labels.csv仅统计这些正标签的Recall和分数，不为全为正例的标签集合计算误导性AP/Precision。该诊断是关联观察：不能据此认定警告样本所有错误均由采样造成，也不修改标签或输入。

ResNet_comparison.csv对照ResNet第一轮单Seed42与VideoMAE三种子汇总；重复次数不对等，不代表ResNet多种子比较。评价单元与31类相同，但预训练来源、维度和时序分辨率不同，属于方案比较。

## 文件、资源与停止

all_seed_results.csv、summary_mean_std.csv、per_class_AP.csv、per_class_AP_stability.csv、length_and_coverage_groups.csv、coverage_warning_labels.csv、ResNet_comparison.csv为统一交付。每轮耗时与GPU/RAM峰值保存于六组独立results.json及统一表；RAM含Workers的求和可能重复计算共享页，PyTorch显存不含桌面/驱动。Seed42资源原样复用，不计入本轮新增六次训练耗时。

Defensive Rearing保持119维输出，但排除训练损失并保护专属参数行，BCE119另作诊断。固定权重Valid指标已与各results.json校验一致。全部工作仅写phase2_5，无Test、额外搜索或后续自动实验。历史保护结果见protection_verification.json。

六组训练及九组固定Checkpoint评价完成，停止等待审核。
"""
    group_summary=[]
    for name in KINDS:
      for split in ('train','valid'):
       for group in sorted({r['group'] for r in groups}):
        a=[r for r in groups if r['model']==name and r['split']==split and r['group']==group]
        group_summary.append(dict(model=name,split=split,group=group,N=a[0]['N'],supported_main_classes=a[0]['supported_main_classes'],supported_main_mAP_mean=float(np.mean([r['supported_main_mAP'] for r in a])),supported_main_mAP_std=float(np.std([r['supported_main_mAP'] for r in a],ddof=1)),Macro_F1_fixed31_mean=float(np.mean([r['Macro_F1_fixed31'] for r in a])),Macro_F1_fixed31_std=float(np.std([r['Macro_F1_fixed31'] for r in a],ddof=1)),BCE118_mean=float(np.mean([r['BCE118'] for r in a]))))
    csvout('group_mean_std.csv',group_summary)
    text+='\n## 数据支持的诊断结论\n'
    for r in summary:
      a=[q for q in rows if q['model']==r['model']]
      unstable=sorted([q for q in stability if q['model']==r['model'] and q['split']=='valid'],key=lambda q:q['AP_std'],reverse=True)[:3]
      labels='、'.join(q['name']+'('+str(q['positives'])+'个正例，AP std='+format(100*q['AP_std'],'.2f')+'个百分点)' for q in unstable)
      text+=f"- {r['head']}：固定Train mAP均值{100*r['Train_mAP_mean']:.2f}%，Valid {100*r['Valid_mAP_mean']:.2f}%，差{100*r['mAP_gap_mean']:.2f}个百分点；F1差{100*r['F1_gap_mean']:.2f}个百分点。最佳mAP轮晚于最低Valid BCE轮{sum(q['best_mAP_minus_min_BCE_epoch']>0 for q in a)}/3次。波动最大的Valid类：{labels}。\n"
    for a,b in (('model2','model1'),('model3','model1'),('model3','model2')):
      pa=[r for r in paired if r['model_a']==a and r['model_b']==b]
      text+=f"- 配对{a}相对{b}：mAP提升{sum(r['Valid_mAP_delta']>0 for r in pa)}/3个种子，F1提升{sum(r['Valid_F1_delta']>0 for r in pa)}/3个种子；逐类提升/下降数量见head_paired_changes.csv。\n"
    text+='\n长度分组诊断（组内AP仅覆盖有正例的主类，各组类别集合不同）：\n'
    for r in group_summary:
      if r['split']=='valid':text+=f"- {r['model']} / {r['group']}：N={r['N']}，支持{r['supported_main_classes']}/31类，条件mAP={100*r['supported_main_mAP_mean']:.2f}±{100*r['supported_main_mAP_std']:.2f}%，固定31类F1={100*r['Macro_F1_fixed31_mean']:.2f}%。\n"
    text+='\n警告标签本身的召回（固定Valid阈值、只计31主类中的警告正例）：\n'
    for name in KINDS:
      a=[r for r in warnings if r['model']==name and r['split']=='valid'];v=[r['main31_missed_sampling_label_recall'] for r in a]
      text+=f"- {name}：{a[0]['main31_warning_positive_pairs']}个警告正标签，Recall={100*np.mean(v):.2f}±{100*np.std(v,ddof=1):.2f}%。样本很少，只作诊断，不据此选模。\n"

    (OUT/'FINAL_PHASE2_5_REPORT.md').write_text(text,encoding='utf-8')
    write_json(OUT/'analysis_complete.json',dict(status='passed',models=9,new_training_runs=6,fixed_Train_Valid_evaluations=18,seconds=time.perf_counter()-start,Test_read=False))
if __name__=='__main__':main()
