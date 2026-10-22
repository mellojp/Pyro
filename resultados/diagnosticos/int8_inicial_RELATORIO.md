# Comparacao FP16 e INT8 no Raspberry Pi 3 B

INT8 foi 2.09 vezes mais rapido no benchmark de rede desta rodada.

| Modelo | Media (ms) | P95 (ms) | FPS da rede |
|---|---:|---:|---:|
| FP16 | 617.8 | 638.5 | 1.62 |
| INT8 | 296.0 | 310.3 | 3.38 |

4 threads, 3 aquecimentos e 20 medicoes. Entrada sintetica; sem captura/preprocessamento/NMS.

| Video/modelo | Quadros | Fumaca 0.25 | Fumaca 0.20 | Fumaca 0.15 | Fogo 0.25 |
|---|---:|---:|---:|---:|---:|
| video1 / FP16 | 56 | 41 | 50 | 56 | 56 |
| video1 / INT8 | 56 | 0 | 0 | 1 | 48 |
| video2 / FP16 | 115 | 30 | 38 | 58 | 114 |
| video2 / INT8 | 115 | 0 | 0 | 3 | 108 |
| video3 / FP16 | 23 | 17 | 23 | 23 | 23 |
| video3 / INT8 | 23 | 0 | 0 | 0 | 22 |

As contagens indicam quadros com pelo menos uma caixa; nao sao recall nem acuracia. Fogo mantido em 0.25 em todas as comparacoes. Os mesmos quadros e saidas sao usados nos 3 limiares de cada modelo.

## Inspecao do arquivo

INT8: 7.372.112 bytes; FP16: 14.151.248 bytes. Modelo original e FP16 identicos aos da rodada anterior por SHA256. INT8 tem I/O UINT8, tensores internos INT8/INT32 e nenhum FLOAT32.

## Limites

- Previa offline com aproximadamente 2 FPS; sem audio.
- Alimentacao disponivel mantida, avisos registrados em cada JSON.
- Execucao sequencial FP16 antes de INT8; temperatura e carga nao controladas.
- Nao foram fornecidos dados/registro de calibracao ou avaliacao rotulada do INT8.
- Videos mostram incendios evidentes; nao medem deteccao precoce nem falsos alarmes em cenas negativas.
- Baixar limiar nao muda o modelo e pode aumentar falsos positivos.

## Arquivos

Videos H.264 terminam em -visual.mp4. Para video3 tambem foi renderizado limiar 0.20. CSVs contem os 3 limiares. JSONs registram configuracoes e tempos. comparar_int8.py pode ser reexecutado na pasta /home/admin/incendio-edge.

## Parecer apos conferir geometria

INT8 atual NAO aprovado: candidato de fumaca do primeiro quadro de video3 tem xywh=(0.3930,0.2590,0,0), enquanto o mesmo indice FP16 tem largura/altura ~0.4354/0.4321. Na saida INT8 desse quadro, 2231 larguras e 2145 alturas sao zero; FP16 nao tem nenhuma. A perda de localizacao nao se resolve apenas baixando confianca. CSVs/contagens/videos principais excluem caixas sem area. Saidas brutas sao preservadas para auditoria.

Recomendacao: FP16 com fogo 0.25 e fumaca 0.20 como experimento. Investigar/reexportar INT8 e sua calibracao, comparando saidas intermediarias e cabeca de deteccao; considerar quantizacao parcial se necessario. Validar em conjunto rotulado e cenas negativas.

O problema de largura/altura zero persistiu ao desativar o XNNPACK; nao foi resolvido trocando para os kernels padrao. Ver diagnostico-sem-xnnpack.json.
