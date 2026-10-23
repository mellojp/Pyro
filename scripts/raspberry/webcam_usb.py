import argparse,json,time,threading,hashlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import cv2,numpy as np
from edge_runtime import Detector
from postprocess import suppress
from hard_negative_collection import HardNegativeCollector

def main():
 p=argparse.ArgumentParser(description='Webcam USB: deteccao no quadro mais recente, sem fila.')
 p.add_argument('--camera',type=int,default=0);p.add_argument('--model',default='modelos/int8_224.tflite');p.add_argument('--seconds',type=float,default=60);p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480);p.add_argument('--camera-fps',type=float,default=30);p.add_argument('--threads',type=int,default=2);p.add_argument('--fire-conf',type=float,default=.25);p.add_argument('--smoke-conf',type=float,default=.15);p.add_argument('--fire-iou',type=float,default=.15);p.add_argument('--smoke-iou',type=float,default=.45);p.add_argument('--preview-port',type=int,default=0);p.add_argument('--containment',type=float,default=0,help='Opcional: .90 suprime caixas de fogo contidas; 0 desativa');p.add_argument('--metrics',default='resultados/camera/webcam-usb-metricas.json');p.add_argument('--collect-dir',help='Pasta nova para candidatos negativos, pendentes de revisao');p.add_argument('--confirm-no-fire-smoke',action='store_true');p.add_argument('--collect-interval',type=float,default=2.0);p.add_argument('--collect-max-images',type=int,default=300);p.add_argument('--collect-min-change',type=float,default=3.0);a=p.parse_args()
 if a.collect_dir and not a.confirm_no_fire_smoke:p.error('Confirme que nao ha fogo nem fumaca com --confirm-no-fire-smoke')
 if a.collect_interval<=0 or a.collect_max_images<1 or a.collect_min_change<0:p.error('Intervalo, limite ou variacao de coleta invalidos')
 if a.collect_dir and Path(a.collect_dir).exists() and any(Path(a.collect_dir).iterdir()):p.error('Use uma pasta vazia ou nova para cada coleta')
 collector=None
 if a.seconds<=0:p.error('--seconds deve ser positivo')
 if a.threads<1:p.error('--threads deve ser positivo')
 if a.containment and not 0<a.containment<=1:p.error('--containment deve ser 0 ou estar em (0,1]')
 if not all(0<x<1 for x in [a.fire_conf,a.smoke_conf,a.fire_iou,a.smoke_iou]):p.error('Confiancas e IoUs devem estar entre 0 e 1')
 cap=cv2.VideoCapture(a.camera,cv2.CAP_V4L2)
 if not cap.isOpened():raise SystemExit(f'Webcam USB /dev/video{a.camera} nao disponivel. Conecte-a e confira v4l2-ctl --list-devices.')
 cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height);cap.set(cv2.CAP_PROP_FPS,a.camera_fps);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
 actual=dict(width=cap.get(3),height=cap.get(4),reported_fps=cap.get(5),fourcc=int(cap.get(6)))
 det=Detector(a.model,a.threads,a.fire_conf,a.smoke_conf)
 for _ in range(3):det.infer(np.zeros((a.height,a.width,3),np.uint8))
 stop=threading.Event();condition=threading.Condition();shared=dict(sequence=0,frame=None,received_at=0.,error=None,captured=0,jpeg=None);http=None
 def capture():
  failures=0
  try:
   while not stop.is_set():
    ok,frame=cap.read();received=time.perf_counter()
    if not ok:
     failures+=1
     if failures>=10:
      with condition:shared['error']='Webcam parou de fornecer quadros';condition.notify_all()
      break
     time.sleep(.02);continue
    failures=0
    with condition:shared.update(sequence=shared['sequence']+1,frame=frame,received_at=received,captured=shared['captured']+1);condition.notify_all()
  finally:cap.release()
 class Preview(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   if self.path!='/stream':
    self.send_response(200);self.send_header('Content-Type','text/html');self.end_headers();self.wfile.write(b'<title>Webcam - deteccoes</title><img src="/stream">');return
   self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers();last=None
   try:
    while not stop.is_set():
     with condition:
      condition.wait_for(lambda:shared['jpeg'] is not last or stop.is_set(),timeout=1);jpeg=shared['jpeg']
     if jpeg is None or jpeg is last:continue
     last=jpeg;self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush()
   except (BrokenPipeError,ConnectionResetError):pass
 if a.preview_port:
  http=ThreadingHTTPServer(('127.0.0.1',a.preview_port),Preview);threading.Thread(target=http.serve_forever,daemon=True).start();print(f'Preview local: http://127.0.0.1:{a.preview_port}',flush=True)
 if a.collect_dir:
  collector=HardNegativeCollector(a.collect_dir,dict(arguments=vars(a),camera_reported=actual,model_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),model_input_shape=det.inp['shape'].tolist()),interval=a.collect_interval,max_images=a.collect_max_images,min_change=a.collect_min_change,confirmed_no_fire_smoke=a.confirm_no_fire_smoke)
  print(f'Coleta de candidatos pendentes de revisao: {a.collect_dir}',flush=True)
 worker=threading.Thread(target=capture,daemon=True);worker.start();start=time.perf_counter();last=0;skipped=0;samples=[];failure=None;last_status=start
 try:
  while time.perf_counter()-start<a.seconds:
   with condition:
    ready=condition.wait_for(lambda:shared['sequence']>last or shared['error'] is not None,timeout=3)
    if shared['error']:raise RuntimeError(shared['error'])
    if not ready:raise RuntimeError('Sem novo quadro da webcam por 3 segundos')
    seq=shared['sequence'];frame=shared['frame'].copy();received=shared['received_at']
   skipped+=max(0,seq-last-1);last=seq;processing_start=time.perf_counter();candidates,times=det.infer(frame);t=time.perf_counter();boxes=suppress(candidates,a.fire_conf,a.smoke_conf,a.fire_iou,a.smoke_iou,containment=a.containment or None);times['nms_ms']=(time.perf_counter()-t)*1000
   detection_ready=time.perf_counter();times['software_frame_age_ms']=(detection_ready-received)*1000;times['detection_processing_ms']=(detection_ready-processing_start)*1000
   if collector:collector.observe(frame,boxes,seq,times)
   if a.preview_port:
    fps=(len(samples)+1)/(time.perf_counter()-start);image=det.draw(frame,boxes,[f'{Path(a.model).stem} | fire {a.fire_conf:.2f} smoke {a.smoke_conf:.2f}',f'Detection {fps:.2f} FPS | collected {collector.saved if collector else 0}',f'Network {times["inference_ms"]:.0f} ms | age {times["software_frame_age_ms"]:.0f} ms']);ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
    if ok:
     with condition:shared['jpeg']=jpeg.tobytes();condition.notify_all()
   times['iteration_ms']=(time.perf_counter()-processing_start)*1000;samples.append(times)
   if time.perf_counter()-last_status>=5:
    elapsed=time.perf_counter()-start;print(f'capture={shared["captured"]/elapsed:.2f} FPS detection={len(samples)/elapsed:.2f} FPS skipped={skipped} inference={times["inference_ms"]:.0f} ms',flush=True);last_status=time.perf_counter()
 except KeyboardInterrupt:pass
 except Exception as exc:failure=str(exc)
 finally:
  end=time.perf_counter();captured_final=shared['captured']
  stop.set()
  with condition:condition.notify_all()
  worker.join(timeout=3)
  if http:http.shutdown();http.server_close()
  elapsed=end-start;summary=dict(arguments=vars(a),camera_reported=actual,wall_seconds=elapsed,captured_frames=captured_final,processed_frames=len(samples),skipped_software_frames=skipped,capture_fps=captured_final/elapsed,detection_fps=len(samples)/elapsed,error=failure,note='Frame age starts when OpenCV read returns; excludes unknown sensor/USB/driver latency. No tracking/interpolation counted as inference.')
  if samples:
   summary['mean_ms']={k:float(np.mean([s[k] for s in samples])) for k in samples[0]};summary['p95_software_frame_age_ms']=float(np.percentile([s['software_frame_age_ms'] for s in samples],95))
  if collector:collector.close();summary['collection']=dict(directory=a.collect_dir,**collector.summary())
  path=Path(a.metrics);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if failure:raise SystemExit(failure)
if __name__=='__main__':main()
