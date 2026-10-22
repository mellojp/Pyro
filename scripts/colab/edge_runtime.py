import time, numpy as np, cv2
try:
 from ai_edge_litert.interpreter import Interpreter
except ImportError:
 from tensorflow.lite.python.interpreter import Interpreter
class Detector:
 def __init__(self,model,threads=4,fire_conf=.25,smoke_conf=.20):
  self.net=Interpreter(model_path=str(model),num_threads=threads);self.net.allocate_tensors();self.inp=self.net.get_input_details()[0];self.out=self.net.get_output_details()[0];self.limits=np.array([fire_conf,smoke_conf],np.float32)
  self.height,self.width=map(int,self.inp['shape'][1:3])
 def infer(self,frame):
  start=time.perf_counter();h,w=frame.shape[:2];r=min(self.width/w,self.height/h);nw,nh=round(w*r),round(h*r);dx,dy=(self.width-nw)/2,(self.height-nh)/2;left,top=round(dx-.1),round(dy-.1)
  image=cv2.copyMakeBorder(cv2.resize(frame,(nw,nh)),top,round(dy+.1),left,round(dx+.1),cv2.BORDER_CONSTANT,value=(114,114,114));rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  if self.inp['dtype']==np.uint8 and abs(self.inp['quantization'][0]-1/255)<1e-8 and self.inp['quantization'][1]==0:x=rgb[None]
  else:
   x=rgb.astype(np.float32)[None]/255
   if np.issubdtype(self.inp['dtype'],np.integer):
    s,z=self.inp['quantization'];lim=np.iinfo(self.inp['dtype']);x=np.clip(np.rint(x/s+z),lim.min,lim.max).astype(self.inp['dtype'])
  self.net.set_tensor(self.inp['index'],x);pre=(time.perf_counter()-start)*1000
  t=time.perf_counter();self.net.invoke();infer=(time.perf_counter()-t)*1000
  t=time.perf_counter();pred=self.net.get_tensor(self.out['index'])[0]
  if np.issubdtype(pred.dtype,np.integer):s,z=self.out['quantization'];pred=(pred.astype(np.float32)-z)*s
  if not np.isfinite(pred).all():raise RuntimeError('Non-finite output')
  scores=pred[:,4:5]*pred[:,5:];cls=np.argmax(scores,axis=1);conf=scores[np.arange(len(scores)),cls];keep=conf>=self.limits[cls];boxes=pred[keep,:4].copy()*np.array([self.width,self.height,self.width,self.height]);conf=conf[keep];cls=cls[keep]
  boxes[:,:2]-=boxes[:,2:]/2
  boxes[:,[0,2]]/=r;boxes[:,[1,3]]/=r;boxes[:,0]-=left/r;boxes[:,1]-=top/r
  x1=np.clip(boxes[:,0],0,w-1);y1=np.clip(boxes[:,1],0,h-1);x2=np.clip(boxes[:,0]+boxes[:,2],0,w-1);y2=np.clip(boxes[:,1]+boxes[:,3],0,h-1)
  valid=(x2>x1)&(y2>y1);boxes=np.stack([x1,y1,x2-x1,y2-y1],axis=1)[valid];conf=conf[valid];cls=cls[valid]
  return (boxes,conf,cls),dict(preprocess_ms=pre,inference_ms=infer,decode_ms=(time.perf_counter()-t)*1000)
 def nms(self,candidates,fire_iou=.30,smoke_iou=.45):
  boxes,conf,cls=candidates;result=[]
  for category,iou in [(0,fire_iou),(1,smoke_iou)]:
   indices=np.flatnonzero(cls==category)
   if not len(indices):continue
   selected=cv2.dnn.NMSBoxes(boxes[indices].tolist(),conf[indices].tolist(),float(self.limits[category]),iou)
   for j in np.asarray(selected).reshape(-1):
    idx=indices[int(j)];x,y,w,h=boxes[idx];result.append([float(x),float(y),float(x+w),float(y+h),float(conf[idx]),category])
  return result
 def draw(self,frame,detections,texts=()):
  image=frame.copy()
  for x1,y1,x2,y2,score,category in detections:
   color=[(30,60,255),(0,210,255)][category];cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),color,2);cv2.putText(image,f'{["fire","smoke"][category]} {score:.2f}',(round(x1),max(20,round(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
  for j,text in enumerate(texts):
   cv2.putText(image,text,(8,22+j*22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3);cv2.putText(image,text,(8,22+j*22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  return image
