
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "false"

import csv
import json
import random
import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf

import sys, pathlib
sys.modules.setdefault("pathlib._local", pathlib)
from models.experimental import attempt_load
from models.tf import TFModel

cfg = json.loads(Path("config_exportacao.json").read_text())
saida = Path(cfg["saida"])
dataset = Path(cfg["dataset"])
rng = random.Random(cfg["seed"])

torch.set_num_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def imagens_split(nomes):
    resultado = []
    for p in sorted(dataset.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensoes:
            continue
        partes = {x.lower() for x in p.relative_to(dataset).parts[:-1]}
        if "images" in partes and partes.intersection(nomes):
            resultado.append(p)
    return resultado

treino = imagens_split({"train"})
validacao = imagens_split({"val", "valid", "validation"})

assert treino, (
    "Não encontrei imagens de treino. Esperava images/train "
    "ou train/images dentro do ZIP do dataset."
)

# Amostra reproduzível apenas do treino.
rng.shuffle(treino)
n_calib = min(cfg["n_calibracao"], len(treino))
calibracao = treino[:n_calib]

# Prefere validação; senão, usa treino não utilizado na calibração.
if validacao:
    rng.shuffle(validacao)
    diagnostico = validacao[:cfg["n_diagnostico"]]
    origem_diagnostico = "validacao"
else:
    diagnostico = treino[n_calib:n_calib + cfg["n_diagnostico"]]
    origem_diagnostico = "treino_fora_da_calibracao"

assert diagnostico, (
    "Não há imagens separadas para diagnóstico. "
    "Reduza N_CALIBRACAO ou inclua a divisão de validação."
)

def preparar(p):
    imagem = cv2.imread(str(p))
    if imagem is None:
        raise RuntimeError(f"Não consegui ler: {p}")
    h, w = imagem.shape[:2]
    escala = min(320 / w, 320 / h)
    nw, nh = round(w * escala), round(h * escala)
    dx, dy = (320 - nw) / 2, (320 - nh) / 2
    imagem = cv2.resize(imagem, (nw, nh))
    imagem = cv2.copyMakeBorder(
        imagem,
        round(dy - .1), round(dy + .1),
        round(dx - .1), round(dx + .1),
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    return imagem.astype(np.float32)[None] / 255.0

# Validar previamente os arquivos da calibração.
for p in calibracao:
    preparar(p)

manifesto = {
    "seed": cfg["seed"],
    "numero_calibracao": len(calibracao),
    "calibracao": [str(p.relative_to(dataset)) for p in calibracao],
    "origem_diagnostico": origem_diagnostico,
    "diagnostico": [str(p.relative_to(dataset)) for p in diagnostico],
    "nota": "Diagnóstico numérico; não substitui avaliação rotulada.",
}
(saida / "imagens_utilizadas.json").write_text(
    json.dumps(manifesto, indent=2)
)


modelo = attempt_load("best.pt", device=torch.device("cpu"), inplace=True, fuse=True)
modelo.eval().float()
nomes_lista = ["fire", "smoke"]
# 6. Carregar os dois arquivos e interpretar corretamente UINT8.
def carregar(nome):
    m = tf.lite.Interpreter(
        model_path=str(saida / nome), num_threads=2
    )
    m.allocate_tensors()
    return m

fp32 = carregar("best-fp32-controle.tflite")
int8 = carregar("best-int8.tflite")

def inferir(m, x):
    i = m.get_input_details()[0]
    o = m.get_output_details()[0]

    if np.issubdtype(i["dtype"], np.integer):
        escala, zero = i["quantization"]
        limites = np.iinfo(i["dtype"])
        x = np.clip(
            np.rint(x / escala + zero), limites.min, limites.max
        ).astype(i["dtype"])
    else:
        x = x.astype(i["dtype"])

    m.set_tensor(i["index"], x)
    m.invoke()
    y = m.get_tensor(o["index"])[0]

    if np.issubdtype(o["dtype"], np.integer):
        escala, zero = o["quantization"]
        y = (y.astype(np.float32) - zero) * escala

    assert y.shape == (6300, 7)
    assert np.isfinite(y).all()
    return y

linhas = []
exemplos = []

for numero, p in enumerate(diagnostico):
    x = preparar(p)

    with torch.no_grad():
        pyt = modelo(
            torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
        )[0][0].cpu().numpy().copy()

    # PyTorch usa pixels; TensorFlow usa coordenadas normalizadas.
    pyt[:, :4] /= 320.0
    # PyTorch: anchor-major; TF v7.0: grid-major em cada escala.
    partes = []
    inicio = 0
    for grade in [40, 20, 10]:
        quantidade = 3 * grade * grade
        bloco = pyt[inicio:inicio + quantidade]
        partes.append(bloco.reshape(3, grade * grade, 7).transpose(1, 0, 2).reshape(-1, 7))
        inicio += quantidade
    pyt = np.concatenate(partes, axis=0)
    controle = inferir(fp32, x)
    quantizado = inferir(int8, x)

    scores = controle[:, 4:5] * controle[:, 5:]
    relevantes = scores.max(axis=1) >= .15
    base_valida = (controle[:, 2] > 0) & (controle[:, 3] > 0)
    mascara = relevantes & base_valida

    invalidas = (
        (quantizado[:, 2] <= 0) | (quantizado[:, 3] <= 0)
    )

    linha = {
        "imagem": str(p.relative_to(dataset)),
        "mae_pytorch_fp32": float(np.abs(pyt - controle).mean()),
        "mae_fp32_int8": float(np.abs(controle - quantizado).mean()),
        "candidatos_relevantes_fp32": int(mascara.sum()),
        "relevantes_com_caixa_invalida_int8": int(
            (mascara & invalidas).sum()
        ),
        "larguras_nao_positivas_int8": int(
            (quantizado[:, 2] <= 0).sum()
        ),
        "alturas_nao_positivas_int8": int(
            (quantizado[:, 3] <= 0).sum()
        ),
    }
    linhas.append(linha)

    indices = np.argsort(scores.max(axis=1))[-10:][::-1]
    exemplos.append({
        "imagem": linha["imagem"],
        "indices": indices.tolist(),
        "pytorch": pyt[indices].tolist(),
        "fp32": controle[indices].tolist(),
        "int8": quantizado[indices].tolist(),
    })

    # Algumas entradas iguais para repetirmos o diagnóstico na placa.
    if numero < 3:
        cv2.imwrite(
            str(saida / f"entrada_diagnostico_{numero}.png"),
            cv2.cvtColor(
                np.rint(x[0] * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
        )

with (saida / "diagnostico.csv").open("w", newline="") as f:
    escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
    escritor.writeheader()
    escritor.writerows(linhas)

(saida / "comparacao_saidas.json").write_text(
    json.dumps(exemplos, indent=2)
)

def estrutura(m):
    i = m.get_input_details()[0]
    o = m.get_output_details()[0]
    return {
        "entrada": {
            "shape": i["shape"].tolist(),
            "dtype": str(i["dtype"]),
            "quantizacao": i["quantization"],
        },
        "saida": {
            "shape": o["shape"].tolist(),
            "dtype": str(o["dtype"]),
            "quantizacao": o["quantization"],
        },
    }

total = sum(d["candidatos_relevantes_fp32"] for d in linhas)
perdidos = sum(
    d["relevantes_com_caixa_invalida_int8"] for d in linhas
)
mae_controle = float(np.mean([
    d["mae_pytorch_fp32"] for d in linhas
]))

resumo = {
    "tensorflow": tf.__version__,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "classes": nomes_lista,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "checkpoint_sha256": hashlib.sha256(
        Path("best.pt").read_bytes()
    ).hexdigest(),
    "numero_calibracao": len(calibracao),
    "numero_diagnostico": len(diagnostico),
    "mae_media_pytorch_fp32": mae_controle,
    "candidatos_relevantes_fp32": total,
    "relevantes_com_caixa_invalida_int8": perdidos,
    "fracao_relevantes_com_caixa_invalida": (
        perdidos / total if total else None
    ),
    "fp32": estrutura(fp32),
    "int8": estrutura(int8),
    "nota": (
        "Triagem numérica, não certificação de qualidade. "
        "Exige avaliação rotulada e benchmark no Raspberry."
    ),
}

if not total:
    resumo["triagem"] = "INCONCLUSIVA: sem candidatos relevantes."
elif perdidos:
    resumo["triagem"] = (
        "ATENÇÃO: ainda há candidatos com dimensões perdidas no INT8."
    )
else:
    resumo["triagem"] = (
        "Não houve perda de dimensões nos candidatos examinados. "
        "Ainda falta validar precisão e localização."
    )

if mae_controle > .001:
    resumo["alerta_controle"] = (
        "Diferença PyTorch/FP32 merece investigação antes de "
        "atribuir as diferenças somente à quantização."
    )

(saida / "resumo.json").write_text(json.dumps(resumo, indent=2))
(saida / "requirements-instalados.txt").write_text(
    subprocess.check_output(
        [os.sys.executable, "-m", "pip", "freeze"], text=True
    )
)

print("\nRESUMO DO DIAGNÓSTICO:")
print(json.dumps(resumo, indent=2))
