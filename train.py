from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='0')
    a=p.parse_args()
    model=YOLO('yolov8n.pt')
    model.train(data=str(Path(a.data).resolve()),epochs=a.epochs,imgsz=640,batch=8,
                patience=20,seed=42,deterministic=True,workers=2,device=a.device,
                project='runs',name='train',plots=True)
    best=YOLO(str(model.trainer.best))
    metrics=best.val(data=a.data,split='test',imgsz=640,batch=8,device=a.device,
                     project='runs',name='test',plots=True)
    print(metrics.results_dict)

if __name__=='__main__': main()
