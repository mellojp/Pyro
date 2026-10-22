"""Amostra o video no tempo original e renderiza deteccoes na cadencia alvo."""
import argparse,json,time,subprocess,hashlib
from pathlib import Path
import cv2,numpy as np
from edge_runtime import Detector
from postprocess import suppress
p=argparse.ArgumentParser();p.add_argument('--fps',type=float,default=6.2);p.add_argument('--smoke-conf',type=float,default=.15);p.add_argument('--fire-iou',type=float,default=.15);p.add_argument('--output',default=None);a=p.parse_args()
assert 0<a.fps<=30 and 0<a.smoke_conf<1 and 0<a.fire_iou<1
tag=f'fogo-nms{round(a.fire_iou*100):03d}_fumaca-conf{round(a.smoke_conf*100):03d}';fps_tag=f'{a.fps:g}'.replace('.','p');root=Path(a.output or f'resultados/videos/simulacoes_{fps_tag}fps/int8_224/{tag}');root.mkdir(parents=True,exist_ok=True)
model=Path('modelos/int8_224.tflite');det=Detector(model,2,.25,a.smoke_conf)
for _ in range(5):det.infer(np.zeros((480,640,3),np.uint8))
all_results=[]
for video in ['video1','video2','video3']:
 cap=cv2.VideoCapture(f'dados/videos/{video}.mp4');assert cap.isOpened();srcfps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(3));height=int(cap.get(4));total=int(cap.get(7));assert srcfps>=a.fps
 dest=root/f'{video}-int8_224-{tag}-{fps_tag}fps.mp4';raw=root/f'{video}-temporario.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(width,height));assert writer.isOpened()
 index=0;emitted=0;rows=[];next_index=0
 while True:
  ok=cap.grab()
  if not ok:break
  current=index;index+=1
  if current<next_index:continue
  ok,frame=cap.retrieve();assert ok
  start=time.perf_counter();cand,metrics=det.infer(frame);t=time.perf_counter();boxes=suppress(cand,fire_conf=.25,smoke_conf=a.smoke_conf,fire_iou=a.fire_iou,smoke_iou=.45);metrics['nms_ms']=(time.perf_counter()-t)*1000
  frame=det.draw(frame,boxes,[f'INT8 224 | simulacao {a.fps:.1f} FPS',f'conf fire .25 smoke {a.smoke_conf:.2f} | NMS {a.fire_iou:.2f}/.45'])
  metrics['processing_ms']=(time.perf_counter()-start)*1000;metrics['source_frame']=current;metrics['source_seconds']=current/srcfps;rows.append(metrics);writer.write(frame);emitted+=1;next_index=round(emitted*srcfps/a.fps)
  if emitted%60==0:print(video,emitted,'quadros',flush=True)
 cap.release();writer.release()
 subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(raw),'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True);raw.unlink()
 result=dict(video=video,source_fps=srcfps,source_frames=total,source_duration_seconds=total/srcfps,output_frames=emitted,output_fps=a.fps,output_duration_seconds=emitted/a.fps,fire_conf=.25,smoke_conf=a.smoke_conf,fire_nms_iou=a.fire_iou,smoke_nms_iou=.45,threads=2,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),mean_processing_ms=float(np.mean([r['processing_ms'] for r in rows])),p95_processing_ms=float(np.percentile([r['processing_ms'] for r in rows],95)),samples=rows,note='Offline cadence simulation: real inference on distinct frames, original scene speed, no frame interpolation. Not a USB camera or wall-clock throughput benchmark; excludes USB, file read and encoding latency.');all_results.append(result)
 (root/f'{video}-metricas.json').write_text(json.dumps(result,indent=2));print('PRONTO',video,emitted,'quadros',flush=True)
(root/'resumo.json').write_text(json.dumps([{k:v for k,v in d.items() if k!='samples'} for d in all_results],indent=2));print('SIMULACAO_CONCLUIDA',flush=True)
