from pathlib import Path
import os,sys,subprocess,zipfile,shutil,json,hashlib,traceback
ROOT=Path('/content/estudo_out');ROOT.mkdir(exist_ok=True);(ROOT/'modelos').mkdir(exist_ok=True)
TOOLS=Path('/content/estudo_tools');REPO=Path('/content/estudo_repo');VENV=Path('/content/estudo_venv');PY=VENV/'bin/python'
DRIVE=Path('/content/drive/MyDrive/resultados_pesquisa_fogo')
def run(args,**kwargs):
 print('RUN',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),check=True,**kwargs)
try:
 assert DRIVE.is_dir(),'Drive nao montado'
 run([sys.executable,'-m','pip','install','-q','uv'])
 run([sys.executable,'-m','uv','python','install','3.10'])
 if not PY.exists():run([sys.executable,'-m','uv','venv','--python','3.10','--seed',VENV])
 run([PY,'-m','pip','install','pip==24.0','setuptools==68.2.2','wheel==0.43.0'])
 run([PY,'-m','pip','install','torch==2.0.1','torchvision==0.15.2','--index-url','https://download.pytorch.org/whl/cpu'])
 run([PY,'-m','pip','install','tensorflow-cpu==2.13.1','keras==2.13.1','numpy==1.24.3','opencv-python-headless==4.8.1.78','pandas==1.5.3','scipy==1.10.1','matplotlib==3.7.5','seaborn==0.12.2','Pillow==10.4.0','PyYAML==6.0.2','requests==2.32.3','tqdm==4.66.5','psutil==6.1.0','GitPython==3.1.43','packaging==24.2','ipython==8.12.3','thop==0.1.1.post2209072238'])
 if not REPO.exists():run(['git','clone','--depth','1','--branch','v7.0','https://github.com/ultralytics/yolov5.git',REPO])
 if (TOOLS/'best.pt').exists():
  shutil.copy2(TOOLS/'best.pt',REPO/'best.pt')
 else:
  with zipfile.ZipFile(DRIVE/'resultados_treino_v4.zip') as z:
   members=[n for n in z.namelist() if n.endswith('weights/best.pt')];assert len(members)==1,members
   with z.open(members[0]) as src,(REPO/'best.pt').open('wb') as dst:shutil.copyfileobj(src,dst)
 assert hashlib.sha256((REPO/'best.pt').read_bytes()).hexdigest()=='08ab8085de112f14caf43780e6b312fa79997bd596fceca93f4ecbb3021712e5','Checkpoint diferente da referencia'
 data=Path('/content/estudo_dataset');data.mkdir(exist_ok=True)
 if not (data/'.extraido').exists():
  with zipfile.ZipFile(DRIVE/'dataset_yolo_processado.zip') as z:z.extractall(data)
  (data/'.extraido').write_text('ok')
 shutil.copy2(TOOLS/'best-int8-v7-tf213.tflite',ROOT/'modelos/int8_320.tflite');shutil.copy2(TOOLS/'best-fp16.tflite',ROOT/'modelos/fp16_320.tflite')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',YOLOv5_AUTOINSTALL='false')
 for phase in ['export','eval']:
  run([PY,'-u',TOOLS/'exportar_resolucoes.py',phase],cwd=REPO,env=env)
 (ROOT/'requirements.txt').write_text(subprocess.check_output([str(PY),'-m','pip','freeze'],text=True));(ROOT/'git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True))
 print('ESTUDO_COLAB_CONCLUIDO',flush=True)
except Exception:
 (ROOT/'erro.txt').write_text(traceback.format_exc());traceback.print_exc();raise
finally:
 for p in TOOLS.glob('*.py'):shutil.copy2(p,ROOT/p.name)
 if Path('/content/estudo_bootstrap.log').exists():shutil.copy2('/content/estudo_bootstrap.log',ROOT/'execucao.log')
 print('PACOTE',shutil.make_archive(str(DRIVE/'estudo_resolucoes_nms'),'zip',ROOT),flush=True)
