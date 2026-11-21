import csv,json,time
from .support import OUT,sha,write_json,verify

def main():
    if (OUT/'FINAL_VERIFICATION.json').exists():raise FileExistsError('Preserve completed delivery')
    a=json.loads((OUT/'analysis_complete.json').read_text());assert a['status']=='passed' and a['fixed_Train_Valid_evaluations']==18
    r=json.loads((OUT/'run_status.json').read_text());assert r['all_complete'] and len(r['models'])==6
    with (OUT/'all_seed_results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==9 and {(x['model'],int(x['seed'])) for x in rows}=={(m,s) for m in ('model1','model2','model3') for s in (42,123,2026)}
    new=[x for x in rows if int(x['seed'])!=42]
    resources=dict(new_training_runs=6,reused_seed42_runs=3,new_training_wall_seconds=sum(float(x['training_seconds']) for x in new),
                   sequential_runner_seconds=r['runner_seconds'],new_GPU_allocated_peak_MiB=max(float(x['GPU_peak_MiB']) for x in new),
                   new_RAM_parent_peak_MiB=max(float(x['RAM_parent_MiB']) for x in new),new_RAM_workers_sum_peak_MiB=max(float(x['RAM_workers_sum_MiB']) for x in new),
                   fixed_checkpoint_analysis_seconds=a['seconds'],Test_read=False)
    write_json(OUT/'resource_summary.json',resources)
    protection=verify()
    p=OUT/'FINAL_PHASE2_5_REPORT.md'
    p.write_text(p.read_text(encoding='utf-8')+f"\n## 综合判断与最终核验\n\nAttention的Valid mAP均值最高，但仅比BiGRU Mean高约0.50个百分点，且其标准差更大；BiGRU Mean的Macro-F1均值更高、跨种子更稳定。以既定mAP为主时保留Attention为主候选，同时保留BiGRU Mean；现有三种子不足以认定两者有显著差异。\n\n三个头的固定Train−Valid mAP差均约53个百分点，支持可重复的严重泛化差距。不能单凭本轮区分标签噪声、背景/物种分布或时序信息缺失的因果贡献；没有据此启动新实验。\n\n新增六组训练累计{resources['new_training_wall_seconds']/60:.2f}分钟，固定权重分析{resources['fixed_checkpoint_analysis_seconds']:.2f}秒。新增训练GPU分配峰值{resources['new_GPU_allocated_peak_MiB']:.2f}MiB，主进程RAM峰值{resources['new_RAM_parent_peak_MiB']:.2f}MiB，含Workers RSS求和峰值{resources['new_RAM_workers_sum_peak_MiB']:.2f}MiB（共享页可能重复计数）。\n\n最终复核{protection['historical_metadata_files']}个历史文件元数据、{protection['historical_hashed_files']}个受保护文件内容SHA无变化。九组固定权重评价、六组新增训练及全部分析完成；停止等待审核。\n",encoding='utf-8')
    files={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and '.runtime' not in p.parts and p.name not in ('artifact_manifest.json','FINAL_VERIFICATION.json')}
    write_json(OUT/'artifact_manifest.json',dict(files=files))
    write_json(OUT/'FINAL_VERIFICATION.json',dict(status='passed',new_training_runs=6,seeds_per_model=3,models=3,fixed_checkpoint_split_evaluations=18,artifact_files=len(files),logical_bytes=sum(x['bytes'] for x in files.values()),artifact_manifest_SHA256=sha(OUT/'artifact_manifest.json'),protection=protection,resources=resources,Test_read=False,next_phase_started=False))
    print(json.dumps(resources),flush=True)
if __name__=='__main__':main()
