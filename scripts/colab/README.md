# Exportação e validação no Colab

As exportações usadas nos experimentos foram feitas com Python 3.10, PyTorch 2.0.1, TensorFlow CPU 2.13.1 e YOLOv5 v7.0. O bootstrap cria um ambiente separado em `/content/estudo_venv`; não instalar essas versões no ambiente Python do Raspberry. O snapshot completo está em `requirements.txt`, e o commit do YOLOv5 em `git_commit.txt`.

## Preparar uma sessão nova

1. Disponibilizar este repositório na sessão do Colab e montar o Google Drive em `/content/drive`.
2. Manter `dataset_yolo_processado.zip` no Drive em `MyDrive/resultados_pesquisa_fogo/`. Esse arquivo não está no Git. Ele deve conter imagens e rótulos YOLO nas divisões de treino e validação; a descoberta aceita `train/images`, `valid/images` ou divisões equivalentes com diretório `images`.
3. Preparar a pasta de ferramentas a partir da raiz do clone:

```bash
mkdir -p /content/estudo_tools
cp scripts/colab/bootstrap.py scripts/colab/exportar_resolucoes.py scripts/colab/avaliar_limiares.py scripts/colab/edge_runtime.py scripts/colab/postprocess.py /content/estudo_tools/
cp treinamento/pesos/best.pt /content/estudo_tools/best.pt
cp modelos/fp16_320.tflite /content/estudo_tools/best-fp16.tflite
cp modelos/int8_320.tflite /content/estudo_tools/best-int8-v7-tf213.tflite
python /content/estudo_tools/bootstrap.py
```

O bootstrap usa o checkpoint copiado para `estudo_tools`; para reproduzir a sessão histórica, também aceita o ZIP original do treino no Drive como alternativa. O hash do checkpoint é conferido antes de carregar. O script instala dependências, clona YOLOv5 v7.0, extrai o dataset, exporta INT8 256/224/192 e avalia junto das referências FP16/INT8 320.

O processo depende de espaço para o dataset extraído e pode demorar. Saídas: `/content/estudo_out`; pacote no Drive: `estudo_resolucoes_nms.zip`. O bootstrap preserva logs e código mesmo se a exportação falhar. Use uma sessão nova ou revise arquivos já existentes antes de executar novamente.

## Protocolo

As novas exportações usam 300 imagens aleatórias do treino para calibração, seed 42. A comparação usa 1.000 imagens da validação, seed 20261021, sem usar o teste. Listas das imagens e métricas preservadas em `resultados/validacao/`. A versão do script avalia confiança de fumaça 0,20 e 0,15; os JSONs informam quais variantes efetivamente foram executadas em cada rodada.

A avaliação usa o mesmo pós-processamento em todas as variantes. O INT8 320 é a referência corrigida anterior, não uma nova exportação desta rodada. O relatório descreve as diferenças em relação ao protocolo original do treinamento.

`avaliar_limiares.py` reutiliza a estrutura criada pelo bootstrap para comparar 224 e 192, salvando resultados de limiares separados. Execute no diretório `/content/estudo_repo`, usando `/content/estudo_venv/bin/python` e `PYTHONPATH=/content/estudo_tools`.

## Diagnóstico anterior da exportação 320

`reexportacao_320/` preserva o exportador e o diagnóstico usados para corrigir a primeira quantização. Seu JSON de configuração contém caminhos da sessão original e deve ser ajustado para outra sessão. O alias de compatibilidade de `pathlib._local` permite abrir o checkpoint no Python 3.10. A comparação PyTorch/TF precisa alinhar a ordem dos candidatos por âncora/célula; o diagnóstico correspondente está em `resultados/diagnosticos/reexportacao_320/`.
