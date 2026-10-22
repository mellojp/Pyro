# NMS e preparacao para webcam USB

## Configuracao inicial proposta

INT8 v7/TF2.13, entrada 320x320, 4 threads, confianca fogo 0.25 e fumaca 0.20. NMS separado por classe: IoU fogo 0.30, fumaca 0.45.

| Video | Quadros | Caixas fogo IoU .45 | IoU .30 | IoU .20 | Quadros com fogo nos 3 ajustes |
|---|---:|---:|---:|---:|---:|
| video1 | 56 | 308 | 176 | 137 | 56 |
| video2 | 115 | 357 | 243 | 216 | 114 |
| video3 | 23 | 174 | 122 | 106 | 23 |

NMS .30 reduziu as caixas de fogo de 839 para 541 (35.5% menos) sem mudar os quadros com fogo nestes videos. Nao comprova recall: nao ha anotacoes. A fumaca foi preservada por ter NMS independente.

NMS e aplicado antes do desenho, sobre candidatos com coordenadas validas e recortadas. IoU menor suprime mais sobreposicoes. Caixas aninhadas com pouco IoU podem permanecer; objetos/focos distintos podem ser indevidamente suprimidos com um ajuste agressivo.

## Perfil de processamento

| Video | Preprocessamento ms | Rede ms | Decodificacao ms | NMS ms | Desenho ms | Processamento ms | FPS de processamento |
|---|---:|---:|---:|---:|---:|---:|---:|
| video1 | 9.0 | 297.8 | 5.7 | 0.8 | 5.2 | 318.5 | 3.14 |
| video2 | 5.7 | 312.9 | 5.6 | 0.7 | 3.5 | 328.4 | 3.04 |
| video3 | 9.7 | 306.5 | 5.7 | 1.1 | 15.0 | 338.0 | 2.96 |

Medicao offline, aproximadamente 2 quadros amostrados por segundo de video. Os tempos nao incluem leitura, exposicao/transporte USB, gravacao ou codificacao. O processamento soma apenas um NMS (.30), nao os tres experimentos. Comparacoes feitas sequencialmente; alimentacao atual mantida e registrada.

10 FPS de deteccao exigem cerca de 100 ms por quadro, incluindo o processamento. A rede atual sozinha consome cerca de 300 ms. Melhorar captura/NMS evita atraso e reduz custo auxiliar, mas nao proporciona o ganho de aproximadamente 3x necessario na rede.

## Experimentos seguintes para 10 FPS

1. Exportar o mesmo checkpoint em 256 e 192, recalibrar INT8 e medir qualidade/velocidade. Reduzir resolucao pode prejudicar fumaca e focos pequenos; nao basta redimensionar o tensor do modelo atual, pois as grades/ancoras tem dimensoes fixas.
2. Se a qualidade cair ou a latencia continuar alta, treinar uma variante menor, prioritariamente YOLOv5n, exportar INT8 e avaliar em 320 e 256. A troca exige treinamento/refino, nao apenas converter o checkpoint atual.
3. Validar mAP/recall por classe, falsos alertas em videos negativos e tempo real ate o alerta com webcam conectada.
Referencia de porte das variantes: https://github.com/ultralytics/yolov5/tree/v7.0#pretrained-checkpoints . FLOPs/parametros menores nao garantem o FPS da placa.

## Aplicacao USB preparada

A webcam nao estava conectada/detectada nesta rodada. O script foi compilado, a interface de comandos conferida e o NMS verificado com caixas sinteticas sobrepostas de classes distintas. O backend de captura/preview ainda precisa de teste com hardware conectado.

No Raspberry:
```bash
cd ~/incendio-edge
.venv/bin/python scripts/webcam_usb.py --camera 0 --seconds 60 --preview-port 8080
```

No Windows, abrir outro terminal para o tunel de preview:
```powershell
ssh -L 8080:127.0.0.1:8080 admin@192.168.18.68
```

Abrir http://localhost:8080 no navegador. O servidor so escuta em 127.0.0.1 na placa; o tunel entrega o preview ao computador. O preview adiciona custo de desenho e JPEG.

Sem preview, para medir deteccao:
```bash
.venv/bin/python scripts/webcam_usb.py --camera 0 --seconds 60
```

A captura usa V4L2 e solicita MJPEG 640x480 a 30 FPS. Sao solicitacoes: o driver pode negociar outra configuracao, registrada no JSON. --camera 0 corresponde a /dev/video0; conferir v4l2-ctl --list-devices.

Uma thread captura continuamente e substitui um unico quadro; a inferencia usa o mais recente e descarta intermediarios. As metricas diferenciam FPS de captura e deteccao. A idade do quadro comeca quando read() retorna, nao no sensor. Nao ha rastreamento/interpolacao contado como inferencia.

Metricas em /home/admin/incendio-edge/resultados/webcam-usb-metricas.json. Sem camera, o script termina com mensagem clara. Nenhum teste ao vivo foi executado nesta rodada.
