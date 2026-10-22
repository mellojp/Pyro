# Pyro

Detecção de fogo e fumaça com YOLOv5, com foco em inferência em CPU no Raspberry Pi 3 Model B. O projeto reúne o checkpoint treinado, exportações TFLite FP16/INT8, medições na placa e simulações de vídeo com caixas.

O treinamento e as exportações foram executados no Google Colab. A integração com uma webcam USB está preparada, mas ainda não foi testada com a câmera física. A meta é aproximadamente 10 FPS, preservando a qualidade das detecções.

## Estado atual

| Variante | mAP50 na validação | FPS de processamento na placa |
|---|---:|---:|
| FP16 320 | 73,9% | — |
| INT8 320 | 46,5% | 3,2 |
| INT8 256 | 48,2% | 4,9 |
| **INT8 224** | **60,0%** | **6,2** |
| INT8 192 | 54,7% | 8,4 |

Qualidade medida nas mesmas 1.000 imagens de validação. FPS dos INT8: 194 quadros dos três vídeos, duas threads, incluindo pré-processamento, rede, NMS e desenho; exclui leitura, captura USB e codificação. O FP16 teve cerca de 1,6 FPS somente da rede em uma rodada anterior, com protocolo diferente. As métricas originais do treinamento não são diretamente comparáveis a esta validação.

O INT8 224 é o candidato atual para equilibrar qualidade e velocidade. Ainda há perda de qualidade em relação ao FP16, portanto a otimização não está encerrada. Perfil atual da webcam: confiança de fogo 0,25 e fumaça 0,15; IoU do NMS de fogo 0,15 e fumaça 0,45; contenção desativada. Também há simulações com NMS de fogo e confiança de fumaça em 0,17.

- [Relatório dos experimentos](resultados/RELATORIO.md)
- [Vídeos organizados por modelo, FPS e parâmetros](resultados/videos/README.md)
- [Comparação em CSV](resultados/comparacao.csv)
- [Apresentação da proposta](documentacao/apresentacao.pdf)

As prévias de 6,2 FPS usam inferência em quadros distintos, sem interpolação, mantendo a velocidade original das cenas. Elas simulam a cadência visual; não medem latência real de uma câmera USB.

## Executar a partir deste repositório

Ambiente validado na placa: Debian 13 arm64, Python 3.13, LiteRT com XNNPACK em CPU. Os comandos abaixo partem da raiz do clone. Os pesos e os três vídeos de entrada estão incluídos.

```bash
git clone https://github.com/mellojp/Pyro.git
cd Pyro
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/raspberry/requirements-referencia.txt

# Simular os três vídeos, a 6,2 FPS, com os dois parâmetros em 0,17:
.venv/bin/python scripts/raspberry/simular_camera_video.py --fire-iou 0.17 --smoke-conf 0.17 --output resultados/videos/minha-simulacao-017

# Quando houver webcam USB disponível:
.venv/bin/python scripts/raspberry/webcam_usb.py --seconds 120 --preview-port 8080
```

O preview é servido apenas em `127.0.0.1`. No PC, crie um túnel e abra `http://localhost:8080`:

```bash
ssh -L 8080:127.0.0.1:8080 USUARIO@IP_DO_RASPBERRY
```

Para experimentar valores intermediários, acrescente `--fire-iou 0.17 --smoke-conf 0.17`. São parâmetros diferentes: IoU maior no NMS permite mais sobreposição; confiança maior filtra escores baixos. Esses ajustes quase não alteram o custo da rede.

Para uma nova rodada de medições, sem sobrescrever os resultados preservados:

```bash
.venv/bin/python scripts/raspberry/estudo_raspberry.py --model modelos/int8_224.tflite --name int8_224_nova_rodada
```

Esse estudo mantém seu protocolo de comparação com confiança de fumaça em 0,20. Os valores de cada rodada estão registrados nos JSONs. O arquivo `perfil_atual.json` documenta os padrões da webcam e não é carregado automaticamente pelo programa.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `modelos/` | Cinco modelos TFLite e descrição das variantes |
| `treinamento/` | Melhor checkpoint, configurações e métricas do treino |
| `dados/videos/` | Três vídeos de entrada para testes visuais |
| `scripts/raspberry/` | Inferência, webcam USB, pós-processamento e benchmarks |
| `scripts/colab/` | Exportação e validação, com dependências próprias |
| `resultados/` | Relatórios, métricas, diagnósticos e prévias |
| `documentacao/` | Proposta e registros das etapas anteriores |

O dataset completo não está incluído; as amostras de calibração e validação estão identificadas em [manifesto.json](resultados/validacao/manifesto.json). Veja as [instruções de exportação no Colab](scripts/colab/README.md). O ZIP completo do treino, logs de execução e inventários de organização ficam apenas no ambiente local; o checkpoint e os resultados necessários estão versionados. Hashes dos modelos: [modelos.json](resultados/manifests/modelos.json).

O [guia do Raspberry](documentacao/RASPBERRY.md) descreve a instalação em `~/incendio-edge`, sincronizada com este repositório e com a mesma estrutura de scripts. Documentos históricos podem citar nomes e caminhos antigos.

## Próximas etapas

- Testar a webcam USB: FPS de captura e detecção, atraso, preview e execução prolongada.
- Investigar a perda de qualidade da quantização e avaliar uma rede menor para aproximar a meta de 10 FPS.
- Confirmar qualidade no conjunto de teste separado e medir falsos alertas em cenas de uso reais.

## Encerrar a placa com segurança

Na sessão SSH, execute `sudo shutdown -h now`. A sessão será encerrada; aguarde o desligamento concluir antes de retirar a alimentação. Para retomar em outro dia, reconecte a alimentação e acesse a placa por SSH.

A implementação de exportação utiliza [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0). As versões e o commit utilizados estão preservados em `scripts/colab/`.
