# INT8 reexportado pelo Colab

Conexao SSH estabelecida usando o Raspberry como cliente Linux. Exportacao e validacao concluidas.

## Ambiente e verificacao

- Python 3.10, TensorFlow 2.13.1, YOLOv5 v7.0; versoes e commit em exportacao/resumo.json.
- Checkpoint original preservado. Alias pathlib._local necessario para abrir o checkpoint nesse Python.
- Comparacao PyTorch/TF alinhada: PyTorch organiza por ancora; TF v7.0 organiza por celula da grade.
- Erro absoluto medio PyTorch/FP32: 1.12783731e-07.
- Candidatos relevantes examinados: 494; dimensoes invalidas no INT8: 0.
- 300 imagens aleatorias do treino para calibracao; 20 da validacao no diagnostico. Manifesto no pacote.

## Raspberry Pi 3 B

Benchmark da rede: 323.8 ms em media, mediana 303.5 ms, p95 453.5 ms, 3.09 FPS. 4 threads; 3 aquecimentos e 20 medicoes.

| Video | Quadros | FP16 media ms | Novo INT8 media ms | FP16 fumaca .20 | Novo INT8 fumaca .20 |
|---|---:|---:|---:|---:|---:|
| video1 | 56 | 622.6 | 295.7 | 50 | 56 |
| video2 | 115 | 622.6 | 298.6 | 38 | 70 |
| video3 | 23 | 620.8 | 296.6 | 23 | 23 |

Fogo mantido em .25. A contagem de quadros com caixa valida de fogo foi a mesma do FP16 nesses videos: 56, 114 e 23.

Mais quadros com caixas NAO demonstra mais acuracia: faltam rotulos e cenas negativas. Previa offline a aproximadamente 2 FPS, sem audio. FPS informado e somente da rede, sem captura, preprocessamento ou NMS.

A alimentacao disponivel foi mantida; estado e temperatura registrados nos JSONs. A referencia FP16 e da rodada anterior, nao uma medicao simultanea.

O novo INT8 passou na triagem de dimensoes e e candidato a avaliacao mais completa; nao foi validado por mAP/recall nem tempo completo ate o alarme.

## Arquivos

- exportacao/: modelos, manifesto, diagnostico numerico, scripts e logs do Colab.
- testes-raspberry/: videos H.264 com sufixo -visual.mp4, quadros, CSVs e resumos.
- resultados_int8_colab_reexport_v7_tf213.zip: pacote original salvo no Drive.
- Modelo na placa: /home/admin/incendio-edge/modelos/best-int8-v7-tf213.tflite.
