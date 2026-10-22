import argparse,csv,gc,json,time,hashlib,subprocess
from pathlib import Path
import cv2,numpy as np
from edge_runtime import Detector
from postprocess import suppress,CONFIGS
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--output',default='resultados/raspberry');p.add_argument('--videos',nargs='+',default=['video1','video2','video3']);a=p.parse_args();root=Path(a.output)/a.name;root.mkdir(parents=True,exist_ok=True)
def status():return {k:subprocess.check_output(['vcgencmd',k],text=True).strip() for k in ['measure_temp','get_throttled']}
bench=[]
for threads in [1,2,4]:
 det=Detector(a.model,threads);dummy=np.zeros((480,640,3),np.uint8)
 for _ in range(3):det.infer(dummy)
 start_status=status();samples=[]
 for _ in range(20):_,timing=det.infer(dummy);samples.append(timing)
 result=dict(threads=threads,mean_network_ms=float(np.mean([s['inference_ms'] for s in samples])),p95_network_ms=float(np.percentile([s['inference_ms'] for s in samples],95)),mean_prepare_decode_ms=float(np.mean([s['preprocess_ms']+s['decode_ms'] for s in samples])),status_before=start_status,status_after=status(),samples=samples)
 bench.append(result);print('BENCHMARK',a.name,json.dumps(result),flush=True);del det;gc.collect()
best=min(bench,key=lambda d:d['mean_network_ms']);(root/'benchmark.json').write_text(json.dumps(dict(model=a.name,sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),best_threads=best['threads'],runs=bench),indent=2));det=Detector(a.model,best['threads'])
for v in a.videos:
 cap=cv2.VideoCapture(f'dados/videos/{v}.mp4');assert cap.isOpened();fps=cap.get(5);step=round(fps/2);w=int(cap.get(3));h=int(cap.get(4));writers={key:cv2.VideoWriter(str(root/f'{v}-{key}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/step,(w,h)) for key in CONFIGS};assert all(wr.isOpened() for wr in writers.values());idx=0;sampled=0;cache={};counts={k:[0,0] for k in CONFIGS};frames={k:[0,0] for k in CONFIGS};timings={k:[] for k in CONFIGS};before=status()
 with (root/f'{v}-deteccoes.csv').open('w',newline='') as f:
  cw=csv.writer(f);cw.writerow(['frame','seconds','config','class','confidence','x1','y1','x2','y2'])
  while True:
   ok,image=cap.read()
   if not ok:break
   current=idx;idx+=1
   if current%step:continue
   candidate,timing=det.infer(image);cache[f'b{current}']=candidate[0].astype(np.float32);cache[f's{current}']=candidate[1];cache[f'c{current}']=candidate[2].astype(np.int8)
   for key,config in CONFIGS.items():
    start=time.perf_counter();boxes=suppress(candidate,**config);nms=(time.perf_counter()-start)*1000;present=set()
    for x1,y1,x2,y2,score,category in boxes:
     counts[key][category]+=1;present.add(category);cw.writerow([current,current/fps,key,['fire','smoke'][category],score,x1,y1,x2,y2])
    for category in present:frames[key][category]+=1
    start=time.perf_counter();annotated=det.draw(image,boxes,[f'{a.name} | {key} | fire .25 smoke .20',f'Network {timing["inference_ms"]:.0f} ms | offline 2 fps']);draw=(time.perf_counter()-start)*1000;times=dict(timing,nms_ms=nms,draw_ms=draw);times['processing_ms']=sum(times.values());timings[key].append(times);writers[key].write(annotated)
    if sampled%20==0:cv2.imwrite(str(root/f'{v}-{key}-frame{current:05d}.jpg'),annotated)
   sampled+=1
   if sampled%20==0:print(a.name,v,sampled,flush=True)
 cap.release()
 for wr in writers.values():wr.release()
 np.savez_compressed(root/f'{v}-candidatos.npz',**cache)
 summary=dict(model=a.name,video=v,sampled_frames=sampled,source_fps=fps,sample_step=step,threads=best['threads'],boxes=counts,frames_with_detection=frames,mean_ms={key:{field:float(np.mean([s[field] for s in values])) for field in values[0]} for key,values in timings.items()},p95_processing_ms={key:float(np.percentile([s['processing_ms'] for s in values],95)) for key,values in timings.items()},status_before=before,status_after=status(),note='Offline sampled frames. FPS computed from processing excludes camera/read/encode. Detection counts are not accuracy.')
 (root/f'{v}-resumo.json').write_text(json.dumps(summary,indent=2));print('VIDEO',a.name,v,json.dumps(summary),flush=True)
print('RASPBERRY_CONCLUIDO',a.name,flush=True)
