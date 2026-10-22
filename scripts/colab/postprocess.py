import cv2,numpy as np
CONFIGS={'nms020':dict(fire_iou=.20,containment=None),'nms015':dict(fire_iou=.15,containment=None),'contencao':dict(fire_iou=.20,containment=.90)}
def suppress(candidates,fire_conf=.25,smoke_conf=.20,fire_iou=.20,smoke_iou=.45,containment=None,area_ratio=.65,score_margin=.05,max_det=300):
 boxes,scores,classes=candidates;selected=[]
 for category,threshold,iou in [(0,fire_conf,fire_iou),(1,smoke_conf,smoke_iou)]:
  indices=np.flatnonzero((classes==category)&(scores>=threshold))
  if not len(indices):continue
  kept=cv2.dnn.NMSBoxes(boxes[indices].tolist(),scores[indices].tolist(),threshold,iou)
  for k in np.asarray(kept).reshape(-1):
   j=indices[int(k)];x,y,w,h=boxes[j]
   if w>0 and h>0:selected.append([float(x),float(y),float(x+w),float(y+h),float(scores[j]),int(category)])
 selected.sort(key=lambda r:r[4],reverse=True);selected=selected[:max_det]
 if containment is None:return selected
 result=[]
 for i,small in enumerate(selected):
  if small[5]!=0:result.append(small);continue
  area=(small[2]-small[0])*(small[3]-small[1]);remove=False
  for j,large in enumerate(selected):
   if i==j or large[5]!=0:continue
   large_area=(large[2]-large[0])*(large[3]-large[1])
   if area>area_ratio*large_area or small[4]>large[4]+score_margin:continue
   intersection=max(0,min(small[2],large[2])-max(small[0],large[0]))*max(0,min(small[3],large[3])-max(small[1],large[1]))
   if area>0 and intersection/area>=containment:remove=True;break
  if not remove:result.append(small)
 return result
