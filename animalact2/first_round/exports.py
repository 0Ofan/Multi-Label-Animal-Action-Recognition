"""First-round result/log/curve exports; numerical evaluation stays unchanged."""
import csv
from html import escape
import json
from pathlib import Path
from .support import owned,write_json


def save_history(directory,history):
    directory=owned(directory)
    write_json(directory/"history.json",history)
    (directory/"history.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in history),encoding="utf-8")
    if history:
        with (directory/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)


def save_metrics(directory,metrics):
    directory=owned(directory);write_json(directory/"metrics.json",metrics)
    rows=metrics["per_class"];main=[r for r in rows if r["main_evaluation"]]
    for name,subset in (("per_class_119.csv",rows),("per_class_main31.csv",main)):
        with (directory/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(subset)
    cm=[dict(class_id=r["class_id"],name=r["name"],TP=r["TP"],FP=r["FP"],TN=r["TN"],FN=r["FN"],Precision=r["Precision"],Recall=r["Recall"],F1=r["F1"],threshold=metrics["threshold"],matrix=json.dumps([[r["TN"],r["FP"]],[r["FN"],r["TP"]]])) for r in main]
    with (directory/"confusion_main31.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(cm[0]));w.writeheader();w.writerows(cm)


def curves(directory,history):
    """Export standalone SVG from real history, no extra plotting dependency."""
    panels=[("BCE training objective (118 supported classes)","train_online_objective","valid_objective"),
            ("BCE119 diagnostic","train_online_BCE119","valid_BCE119"),
            ("Fixed31 mAP","train_online_mAP","valid_mAP")]
    items=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="960" viewBox="0 0 1100 960"><rect width="1100" height="960" fill="white"/>']
    for panel,(title,a,b) in enumerate(panels):
        top=panel*320;left=85;width=950;height=220
        values=[float(r[k]) for r in history for k in (a,b)]
        lo,hi=min(values),max(values);span=max(hi-lo,1e-6);lo-=span*.08;hi+=span*.08
        items.append(f'<text x="85" y="{top+26}" font-family="sans-serif" font-size="18">{escape(title)}</text>')
        for i in range(5):
            y=top+50+height*i/4;value=hi-(hi-lo)*i/4
            items.append(f'<path d="M85,{y}H1035" stroke="#e2e8f0"/><text x="18" y="{y+5}" font-family="sans-serif" font-size="12">{value:.4f}</text>')
        for key,color in ((a,"#2563eb"),(b,"#dc2626")):
            points=" ".join(f"{left+(r['epoch']-1)/max(1,len(history)-1)*width:.2f},{top+50+(hi-r[key])/(hi-lo)*height:.2f}" for r in history)
            items.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        items.append(f'<text x="85" y="{top+298}" font-family="sans-serif" font-size="12" fill="#2563eb">Train online (mixed states)</text><text x="400" y="{top+298}" font-family="sans-serif" font-size="12" fill="#dc2626">Valid end-of-epoch checkpoint</text>')
    items.append('</svg>');(owned(directory)/"training_curves.svg").write_text("".join(items),encoding="utf-8")
