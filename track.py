import os
from pathlib import Path
R=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(R/'runtime_config')
os.environ['YOLO_AUTOINSTALL']='false'
import cv2, json, csv, shutil, subprocess, sys
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter, defaultdict, deque
import imageio_ffmpeg
import argparse
p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--model',default=str(R/'best.pt'))
p.add_argument('--output',default=str(R/'runs/audio3'))
p.add_argument('--preview',action='store_true')
args=p.parse_args()
SRC=Path(args.source)
OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(SRC)); fps=cap.get(5); total=int(cap.get(7))
assert fps>0 and total/fps>=5, (fps,total)
print('video',fps,total,total/fps,flush=True)
model=YOLO(args.model)
if '--preview' in sys.argv:
    montage=Image.new('RGB',(1200,450),'white')
    for i,t in enumerate([0,2,4]):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read(); assert ok
        result=model.predict(frame,imgsz=1280,conf=0.1,device=0,verbose=False)[0]
        print(t,[(model.names[int(c)],round(float(s),3)) for c,s in zip(result.boxes.cls,result.boxes.conf)],flush=True)
        im=Image.fromarray(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB)); im.thumbnail((400,450)); montage.paste(im,(i*400,0))
    montage.save(OUT/'preview.jpg'); cap.release(); sys.exit()
cfg=OUT/'bytetrack.yaml'
cfg.write_text('tracker_type: bytetrack\ntrack_high_thresh: 0.4\ntrack_low_thresh: 0.05\nnew_track_thresh: 0.4\ntrack_buffer: 30\nmatch_thresh: 0.8\nfuse_score: True\n')
width,height=int(cap.get(3)),int(cap.get(4)); n=round(fps*5)
writer=cv2.VideoWriter(str(OUT/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height)); assert writer.isOpened()
rows=[]; history=defaultdict(lambda:deque(maxlen=45)); shots=[]
try:
    for index in range(n):
        ok,frame=cap.read(); assert ok
        result=model.track(frame,persist=True,tracker=str(cfg),imgsz=1280,conf=0.4,device=0,verbose=False)[0]
        canvas=result.plot(); boxes=result.boxes
        if boxes.id is not None:
            for box,ident,cls,score in zip(boxes.xywh.cpu().tolist(),boxes.id.int().cpu().tolist(),boxes.cls.int().cpu().tolist(),boxes.conf.cpu().tolist()):
                x,y,w,h=box; history[ident].append((round(x),round(y)))
                cv2.polylines(canvas,[np.array(history[ident],dtype=np.int32).reshape(-1,1,2)],False,(0,255,255),2)
                rows.append([index,index/fps,ident,model.names[cls],score,x,y,w,h])
        writer.write(canvas)
        if index in [0,round(fps*2),round(fps*4)]:
            shots.append(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)))
        if index%30==0: print('frame',index,'/',n,flush=True)
finally:
    cap.release(); writer.release()
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(OUT/'raw.mp4'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(OUT/'audio3_tracking_5s.mp4')],check=True,capture_output=True)
with (OUT/'tracks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['frame','seconds','track_id','class','confidence','cx','cy','w','h']); w.writerows(rows)
summary=dict(source=str(SRC),frames=n,fps=fps,duration=n/fps,imgsz=1280,conf=0.4,classes={name:dict(frames=len({r[0] for r in rows if r[3]==name}),ids=sorted({r[2] for r in rows if r[3]==name})) for name in model.names.values()})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
montage=Image.new('RGB',(1200,450),'white')
for i,im in enumerate(shots): im.thumbnail((400,450)); montage.paste(im,(400*i,0))
montage.save(OUT/'tracking_preview.jpg')
print(json.dumps(summary,ensure_ascii=False),flush=True)
