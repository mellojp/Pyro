# Vídeos por configuração

As duas simulações usam INT8 224, duas threads e reprodução fixa de 6,2 FPS, mantendo a velocidade original da cena. Cada quadro exibido recebe uma inferência; não há interpolação. Confiança de fogo fixa em 0,25, NMS de fumaça fixo em 0,45 e contenção desativada.

| Teste | NMS de fogo (IoU) | Confiança da fumaça | Vídeos |
|---|---:|---:|---|
| Base | 0,15 | 0,15 | [1](simulacoes_6p2fps/int8_224/fogo-nms015_fumaca-conf015/video1-int8_224-fogo-nms015_fumaca-conf015-6p2fps.mp4) · [2](simulacoes_6p2fps/int8_224/fogo-nms015_fumaca-conf015/video2-int8_224-fogo-nms015_fumaca-conf015-6p2fps.mp4) · [3](simulacoes_6p2fps/int8_224/fogo-nms015_fumaca-conf015/video3-int8_224-fogo-nms015_fumaca-conf015-6p2fps.mp4) |
| Novo | 0,17 | 0,17 | [1](simulacoes_6p2fps/int8_224/fogo-nms017_fumaca-conf017/video1-int8_224-fogo-nms017_fumaca-conf017-6p2fps.mp4) · [2](simulacoes_6p2fps/int8_224/fogo-nms017_fumaca-conf017/video2-int8_224-fogo-nms017_fumaca-conf017-6p2fps.mp4) · [3](simulacoes_6p2fps/int8_224/fogo-nms017_fumaca-conf017/video3-int8_224-fogo-nms017_fumaca-conf017-6p2fps.mp4) |

Os índices dos quadros de origem são iguais entre as simulações para facilitar a comparação. Aumentar NMS IoU permite mais sobreposição de fogo; aumentar confiança da fumaça filtra escores baixos. A simulação não reproduz o atraso adicional nem as variações de cadência de uma webcam USB real. Não é avaliação quantitativa de precisão, pois os vídeos não estão rotulados.

## Estrutura

- `simulacoes_6p2fps/int8_224/`: comparação atual, uma pasta por par de parâmetros. Cada pasta guarda os três MP4 e seus metadados.
- `comparacoes_2fps/`: experimentos anteriores, separados por modelo e configuração. As pastas `quadros/` guardam imagens desse mesmo teste.

No nome das pastas, `fogo-nms017` significa IoU do NMS de fogo 0,17; `fumaca-conf017` significa confiança de fumaça 0,17; `contencao090` identifica o filtro adicional de contenção 0,90.

## Comparações anteriores a 2 FPS

| Modelo | NMS fogo | Confiança fumaça | Contenção | Vídeos disponíveis |
|---|---:|---:|---|---|
| INT8 224 | 0,15 | 0,20 | Desativada | 1, 2 e 3 |
| INT8 224 | 0,20 | 0,20 | 0,90 | 3 |
| INT8 192 | 0,15 | 0,20 | Desativada | 3 |
| INT8 320 | 0,20 | 0,20 | Desativada | 3 |
| INT8 320 | 0,20 | 0,20 | 0,90 | 3 |

Há também quadros de outras configurações de NMS, preservados nas respectivas pastas. Compare testes dentro da mesma cadência para julgar fluidez; as prévias de 2 FPS foram feitas para inspeção de caixas.
