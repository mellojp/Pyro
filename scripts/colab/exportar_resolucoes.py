import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',YOLOv5_AUTOINSTALL='false')
import sys,json,random,shutil,csv,gc,hashlib
from pathlib import Path
import cv2,numpy as np,torch,tensorflow as tf,pathlib
sys.modules.setdefault('pathlib._local',pathlib)
sys.path.insert(0,str(Path.cwd()))
from models.experimental import attempt_load
from models.tf import TFModel
from edge_runtime import Detector
from postprocess import suppress,CONFIGS
OPCONFIGS=dict(CONFIGS)
OPCONFIGS.update({key+"_smoke015":dict(value,smoke_conf=.15) for key,value in CONFIGS.items()})
from val import process_batch
from utils.metrics import ap_per_class
ROOT=Path('/content/estudo_out');ROOT.mkdir(exist_ok=True);(ROOT/'modelos').mkdir(exist_ok=True)
D=Path('/content/estudo_dataset');ext={'.jpg','.jpeg','.png','.bmp','.webp'}
def images(split):return sorted(p for p in D.rglob('*') if p.is_file() and p.suffix.lower() in ext and 'images' in p.parts and set(p.parts)&set(split))
train=images(['train']);valid=images(['val','valid','validation']);assert train and valid,'Treino/validacao nao encontrados'
valid_total=len(valid)
random.Random(20261021).shuffle(valid);valid=valid[:1000]
rng=random.Random(42);rng.shuffle(train);cal=train[:300];assert len(cal)>=100
(ROOT/'manifesto.json').write_text(json.dumps(dict(seed=42,validation_seed=20261021,validacao_total=valid_total,validacao_amostra=len(valid),calibracao=[str(p.relative_to(D)) for p in cal],validacao=[str(p.relative_to(D)) for p in valid],testes_usados=False),indent=2))
torch.set_num_threads(2);tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
def prepare(p,size):
 im=cv2.imread(str(p));assert im is not None,str(p);h,w=im.shape[:2];r=min(size/w,size/h);nw,nh=round(w*r),round(h*r);dx,dy=(size-nw)/2,(size-nh)/2
 im=cv2.copyMakeBorder(cv2.resize(im,(nw,nh)),round(dy-.1),round(dy+.1),round(dx-.1),round(dx+.1),cv2.BORDER_CONSTANT,value=(114,114,114));return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[None].astype(np.float32)/255
if sys.argv[1]=='export':
 model=attempt_load('best.pt',device=torch.device('cpu'),inplace=True,fuse=True);model.eval().float()
 assert list(model.names.values())==['fire','smoke'] if isinstance(model.names,dict) else model.names==['fire','smoke']
 for size in [256,224,192]:
  dest=ROOT/f'modelos/int8_{size}.tflite'
  print('EXPORTANDO',size,flush=True);tf.keras.backend.clear_session()
  tm=TFModel(cfg=model.yaml,model=model,nc=2,imgsz=(size,size));tm.predict(tf.zeros((1,size,size,3)));inp=tf.keras.Input(shape=(size,size,3),batch_size=1);out=tm.predict(inp)
  if isinstance(out,(list,tuple)):out=out[0]
  km=tf.keras.Model(inp,out);km.trainable=False
  converter=tf.lite.TFLiteConverter.from_keras_model(km);converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.uint8;converter.inference_output_type=tf.uint8
  converter.representative_dataset=lambda:[ [prepare(p,size)] for p in cal ]
  dest.write_bytes(converter.convert());print('MODELO_PRONTO',size,dest.stat().st_size,flush=True)
  del converter,km,tm;gc.collect()
 print('EXPORTACOES_CONCLUIDAS',flush=True)
else:
 results=[];iouv=torch.linspace(.5,.95,10)
 def labels(p,h,w):
  parts=list(p.parts);parts[parts.index('images')]='labels';lp=Path(*parts).with_suffix('.txt');rows=[]
  if lp.exists():
   for line in lp.read_text().splitlines():
    if not line.strip():continue
    c,x,y,bw,bh=map(float,line.split());assert c in [0,1];rows.append([c,(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
  return np.asarray(rows,np.float32).reshape(-1,5)
 models=[('fp16_320',ROOT/'modelos/fp16_320.tflite')]+[(f'int8_{size}',ROOT/f'modelos/int8_{size}.tflite') for size in [320,256,224,192]]
 for name,path in models:
  print('AVALIANDO',name,'imagens',len(valid),flush=True);det=Detector(path,2,.001,.001);stats=[];operating={key:dict(tp=[0,0],fp=[0,0],fn=[0,0],boxes=[0,0]) for key in OPCONFIGS};targets=np.zeros(2,int)
  for number,p in enumerate(valid):
   image=cv2.imread(str(p));assert image is not None;h,w=image.shape[:2];gt=labels(p,h,w);targets+=np.bincount(gt[:,0].astype(int),minlength=2);candidate,_=det.infer(image)
   apdet=np.asarray(suppress(candidate,.001,.001,.6,.6),np.float32).reshape(-1,6);correct=process_batch(torch.from_numpy(apdet),torch.from_numpy(gt),iouv).numpy();stats.append((correct,apdet[:,4],apdet[:,5],gt[:,0]))
   for key,config in OPCONFIGS.items():
    pred=np.asarray(suppress(candidate,**config),np.float32).reshape(-1,6);ok=process_batch(torch.from_numpy(pred),torch.from_numpy(gt),iouv[:1]).numpy()[:,0]
    for c in [0,1]:
     count=int(np.sum(pred[:,5]==c));tp=int(np.sum(ok&(pred[:,5]==c)));operating[key]['tp'][c]+=tp;operating[key]['fp'][c]+=count-tp;operating[key]['fn'][c]+=int(np.sum(gt[:,0]==c))-tp;operating[key]['boxes'][c]+=count
   if (number+1)%100==0:print(name,number+1,'/',len(valid),flush=True)
  allstats=[np.concatenate([s[k] for s in stats],axis=0) for k in range(4)];perclass={}
  if allstats[0].any():
   tp,fp,pr,rc,f1,ap,classes=ap_per_class(*allstats,plot=False,names={0:'fire',1:'smoke'})
   for j,c in enumerate(classes):perclass[['fire','smoke'][int(c)]]=dict(ap50=float(ap[j,0]),ap50_95=float(ap[j].mean()),precision_best_f1=float(pr[j]),recall_best_f1=float(rc[j]),targets=int(targets[c]))
  for c,label in enumerate(['fire','smoke']):
   perclass.setdefault(label,dict(ap50=0.,ap50_95=0.,precision_best_f1=0.,recall_best_f1=0.,targets=int(targets[c])))
  for key,values in operating.items():
   values['precision']=[values['tp'][c]/(values['tp'][c]+values['fp'][c]) if values['tp'][c]+values['fp'][c] else 0. for c in [0,1]];values['recall']=[values['tp'][c]/(values['tp'][c]+values['fn'][c]) if values['tp'][c]+values['fn'][c] else 0. for c in [0,1]]
  result=dict(model=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,validation_images=len(valid),class_order=['fire','smoke'],perclass=perclass,map50=float(np.mean([d['ap50'] for d in perclass.values()])),map50_95=float(np.mean([d['ap50_95'] for d in perclass.values()])),operating=operating,protocol='AP: best-class predictions conf .001, NMS .60, max_det300, IoU .50:.95. Operating P/R: fire .25 smoke .20 (or .15 in *_smoke015), IoU match .50, all variants share one inference. Validation used for model selection; test untouched.')
  (ROOT/f'{name}-validacao.json').write_text(json.dumps(result,indent=2));results.append(result);(ROOT/'validacao_completa.json').write_text(json.dumps(results,indent=2));print('RESULTADO',name,'mAP50',result['map50'],'operating',operating,flush=True)
  del det;gc.collect()
 print('VALIDACAO_CONCLUIDA',flush=True)
