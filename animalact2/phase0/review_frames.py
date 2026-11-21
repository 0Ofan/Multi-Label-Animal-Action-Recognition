"""Small human-readable source checks only; never a model input cache."""
import csv,json,sys
from PIL import Image,ImageDraw,ImageFont
import cv2,numpy as np
from audit_dataset import ROOT,OUT,DATA,guard,read,save


def frames(row,positions):
    cap=cv2.VideoCapture(str(DATA/row["video_path"]));result={}
    try:
        wanted=set(positions)
        for index in range(max(wanted)+1):
            if not cap.grab():raise RuntimeError("Read failed")
            if index in wanted:
                ok,bgr=cap.retrieve();assert ok
                result[index]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    return [result[i] for i in positions]


def main():
    sys.addaudithook(guard)
    rows=read(DATA/"manifest.json")["samples"];meta={r["video_id"]:r for r in rows}
    with (OUT/"evidence/near_duplicate_candidates.csv").open(encoding="utf-8-sig") as f:candidates=[r for r in csv.DictReader(f) if r["cross_split"]=="True" and "test" not in (r["left_split"],r["right_split"])]
    candidates.sort(key=lambda r:sum(map(int,r["hamming_distances"].split(";"))))
    selected=[];seen=set()
    for r in candidates:
        suffix=r["left"][-3:]+r["right"][-3:]
        if suffix in seen:continue
        seen.add(suffix);selected.append(r)
        if len(selected)==4:break
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",17)
    image=Image.new("RGB",(1000,4*440+50),"white");draw=ImageDraw.Draw(image)
    draw.text((10,10),"Perceptual similarity candidates: source-frame review, not confirmed recording identity",font=font,fill="black")
    records=[]
    for pair_index,pair in enumerate(selected):
        arrays=[]
        for side_index,key in enumerate(("left","right")):
            row=meta[pair[key]];positions=np.rint(np.linspace(0,row["nb_frames"]-1,3)).astype(int).tolist();images=frames(row,positions)
            ann=read(DATA/row["annotation_path"]);labels=sorted({a["action"] for obj in ann["annotations"] for a in obj["actions"]})
            top=45+440*pair_index+220*side_index
            draw.text((10,top),f"{row['split'].upper()} {row['video_id']} | {', '.join(labels)} | {row['duration']:.3f}s",font=font,fill="#225588")
            for col,picture in enumerate(images):
                arrays.append(np.asarray(picture))
                picture.thumbnail((320,180));image.paste(picture,(10+330*col,top+29))
        differences=[float(np.abs(arrays[i].astype(float)-arrays[i+3].astype(float)).mean()) for i in range(3)]
        records.append({**pair,"reviewed_frame_positions":"first/middle/last","RGB_mean_absolute_differences":differences,"manual_status":"pending visual review"})
    image.save(OUT/"evidence/near_duplicate_review.jpg",quality=93)
    save("evidence/near_duplicate_review_records.json",records)
    # Source-only preprocessing illustration; no feature/model computation.
    ids=["AKFJYBGG","CLZLJTLI","HCTTCVRM"]
    image=Image.new("RGB",(1050,3*300+50),"white");draw=ImageDraw.Draw(image)
    draw.text((10,10),"Raw frame | old VideoMAE center crop | proposed full-frame short-edge224 (design only)",font=font,fill="black")
    for ri,vid in enumerate(ids):
        row=meta[vid];ann=read(DATA/row["annotation_path"]);index=int(np.rint(np.linspace(0,row["nb_frames"]-1,16))[7])
        raw=frames(row,[index])[0];w,h=raw.size;resize=raw.resize((int(224*w/min(w,h)),int(224*h/min(w,h))),Image.Resampling.BILINEAR)
        crop=resize.crop(((resize.width-224)//2,(resize.height-224)//2,(resize.width+224)//2,(resize.height+224)//2))
        top=45+300*ri;draw.text((10,top),f"{row['split']} {vid} frame{index}/{row['nb_frames']} | time={index/row['fps']:.4f}s",font=font,fill="#225588")
        for col,picture in enumerate((raw,crop,resize)):
            picture=picture.copy();picture.thumbnail((330,225));image.paste(picture,(10+350*col,top+30))
        active=[a["action"] for obj in ann["annotations"] for a in obj["actions"] if a["start"]<=index/row["fps"]<a["end"]]
        draw.text((10,top+266),"Active annotations at this frame: "+", ".join(sorted(set(active))),font=font,fill="black")
    image.save(OUT/"evidence/preprocessing_source_review.jpg",quality=93)
    print("Two small source-review sheets saved; no training samples generated.")


if __name__=="__main__":main()
