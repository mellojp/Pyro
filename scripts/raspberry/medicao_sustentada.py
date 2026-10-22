import sys,time,json,subprocess
sys.path.insert(0,'scripts')
import cv2,numpy as np
from edge_runtime import Detector
from postprocess import suppress
result=[]
cap=cv2.VideoCapture('dados/videos/video3.mp4');ok,frame=cap.read();cap.release();assert ok
frame=cv2.resize(frame,(640,480))
for size in [224,192]:
 det=Detector(f'modelos/int8_{size}.tflite',2)
 for _ in range(5):det.infer(frame)
 rows=[];before=subprocess.getoutput('vcgencmd get_throttled')
 for i in range(150):
  start=time.perf_counter();cand,t=det.infer(frame);boxes=suppress(cand,fire_iou=.15);det.draw(frame,boxes);t['processing_ms']=(time.perf_counter()-start)*1000;rows.append(t)
 result.append(dict(model=f'int8_{size}',threads=2,runs=150,input='same video3 first frame resized 640x480, no read/encode in timing',mean_ms={k:float(np.mean([r[k] for r in rows])) for k in rows[0]},p95_processing_ms=float(np.percentile([r['processing_ms'] for r in rows],95)),before=before,after=subprocess.getoutput('vcgencmd get_throttled'),samples=rows))
 print(size,result[-1]['mean_ms'],flush=True)
open('resultados/raspberry/sustentado.json','w').write(json.dumps(result,indent=2))
