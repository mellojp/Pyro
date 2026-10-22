# Parecer dos experimentos de resolução e caixas

O INT8 224 foi o melhor compromisso entre qualidade medida e velocidade entre as variantes INT8 avaliadas. O INT8 192 se aproxima mais da meta de 10 FPS. Porém a validação revelou perda de qualidade relevante em todos os INT8: não considero a otimização encerrada para uma aplicação que exige detecções precisas. Eles são candidatos para protótipo e testes da câmera.

## Qualidade e latência

| Modelo | AP50 fogo | AP50 fumaça | mAP50 | Rede média nos vídeos | Processamento médio | FPS de processamento |
|---|---:|---:|---:|---:|---:|---:|
| FP16 320 | 66,3% | 81,5% | 73,9% | — | — | — |
| INT8 320 | 34,7% | 58,4% | 46,5% | 300,2 ms | 311,5 ms | 3,21 |
| INT8 256 | 35,0% | 61,3% | 48,2% | 193,2 ms | 203,2 ms | 4,92 |
| INT8 224 | 50,4% | 69,6% | 60,0% | 151,1 ms | 160,3 ms | 6,24 |
| INT8 192 | 45,7% | 63,7% | 54,7% | 110,3 ms | 118,6 ms | 8,43 |

A referência FP16 na placa, medida anteriormente, foi cerca de 623 ms somente da rede com quatro threads; não foi repetida nesta rodada. Não comparar esse tempo como se fosse o mesmo protocolo completo dos INT8.

Qualidade: 1.000 imagens de uma divisão de validação com 2.123 imagens, amostra aleatória seed 20261021; 268 objetos de fogo e 1.139 de fumaça. As mesmas imagens e o mesmo código foram usados em todos os modelos. AP usa melhor classe por candidato, confiança mínima 0,001, NMS 0,60, máximo 300 detecções e correspondência IoU 0,50:0,95. Portanto há diferenças de protocolo em relação ao `val.py` padrão e à avaliação original do treino. O conjunto de teste não foi usado. A amostra serviu para selecionar variantes; falta confirmação independente no teste e em cenas de uso reais.

As três novas resoluções foram exportadas do mesmo checkpoint usando TF 2.13.1, YOLOv5 v7.0 e 300 imagens do treino para calibração, seed 42. O INT8 320 é a exportação corrigida anterior; não foi reexportado nesta rodada. Assim, não atribuir toda a diferença entre variantes exclusivamente à resolução. Checkpoint, dependências e lista de imagens estão preservados nos arquivos de reprodução e manifestos.

Latência: 194 quadros amostrados nos três vídeos, duas threads, uma inferência por quadro compartilhada pelos três pós-processamentos. Média ponderada pelo número de quadros; processamento soma pré-processamento, rede, decodificação, NMS 0,15 e desenho. Não inclui leitura do vídeo, sensor, USB, JPEG/HTTP nem gravação. Os vídeos não têm rótulos: contagens de caixas e quadros positivos não são acurácia.

No benchmark curto (três aquecimentos e 20 inferências por ajuste), duas threads venceram uma e quatro em todas as resoluções. Exemplos: 320 com 290,5 ms em duas e 323,5 ms em quatro; 224 com 145,0 e 177,2 ms. O benchmark inicial de 192 coincidiu parcialmente com transferência de arquivos; por isso foi feita também medição contínua sem transferências: 150 inferências, cinco aquecimentos, mesmo quadro redimensionado a 640x480. Nessa medição, 224 teve 151,5 ms de processamento (6,60 FPS) e 192 teve 111,1 ms (9,00 FPS). Isso verifica o potencial de processamento, mas não equivale a vídeo variado nem a câmera real. Amostras e p95 estão em `raspberry/sustentado.json`.

A alimentação disponível foi mantida. Os logs registram temperatura e flags da placa; na rodada contínua, `get_throttled` permaneceu em `0x50000`, com histórico registrado. Os tempos descrevem esta placa nestas condições.

## Caixas sobrepostas

O NMS por IoU não elimina necessariamente uma caixa pequena dentro de outra grande: a interseção pode ser grande em relação à pequena e pequena em relação à união. Testamos IoU de fogo 0,20, 0,15 e uma regra adicional de contenção; fumaça manteve NMS independente 0,45.

A contenção só atua em fogo: interseção/área menor >= 0,90, área menor <= 65% da maior e confiança menor <= confiança maior + 0,05. Não é agrupamento de todos os focos numa única caixa.

| INT8 224, validação | TP fogo | FP fogo | FN fogo | Precisão fogo | Recall fogo |
|---|---:|---:|---:|---:|---:|
| NMS 0,20 | 150 | 102 | 118 | 59,5% | 56,0% |
| NMS 0,15 | 149 | 98 | 119 | 60,3% | 55,6% |
| NMS 0,20 + contenção | 149 | 100 | 119 | 59,8% | 55,6% |

No vídeo 3, INT8 320 passou de 106 caixas de fogo com NMS 0,20 para 105 com 0,15 e 96 com contenção, mantendo os 23 quadros com fogo. Isso não prova preservação de todos os focos. Na validação, contenção perdeu dois TP de fogo em 320 e cinco em 192. Portanto deixei NMS 0,15 como ajuste inicial e contenção opcional. Ainda podem existir sobreposições, especialmente em fumaça; não foi aplicado filtro agressivo de fumaça.

A redução de caixas tem efeito visual, mas praticamente não resolve o tempo: o NMS consome menos de 1 ms em média; a rede consome aproximadamente 93–96% do processamento dos INT8. Reduzir mais o IoU tende a trocar recall por aparência limpa, sem resolver a causa dos falsos positivos.

## Confiança de fumaça

| Modelo | Confiança | Precisão fumaça | Recall fumaça | TP | FP |
|---|---:|---:|---:|---:|---:|
| INT8 224 | 0,20 | 73,3% | 68,6% | 781 | 284 |
| INT8 224 | 0,15 | 68,2% | 71,0% | 809 | 377 |
| INT8 192 | 0,20 | 71,7% | 64,2% | 731 | 288 |
| INT8 192 | 0,15 | 67,8% | 67,3% | 767 | 365 |

Correspondência IoU 0,50, fogo com confiança 0,25 e NMS 0,15. Manter 0,20 como padrão para conter falsos positivos; usar 0,15 quando a prioridade for sensibilidade e acompanhar falsos alertas. Não confundir confiança do modelo com probabilidade calibrada de incêndio.

## Decisão e trabalho restante

1. Para iniciar a integração USB: INT8 224, duas threads, fogo 0,25, fumaça 0,20, NMS fogo 0,15/fumaça 0,45, contenção desativada. Para buscar 8–9 FPS de processamento: INT8 192, aceitando menor qualidade. São configurações provisórias.
2. Para melhorar velocidade sem acumular atraso: captura em uma thread com apenas o quadro mais recente; evitar filas, gravação simultânea e preview excessivo; solicitar MJPEG 640x480 na webcam e conferir o formato realmente negociado. O script já implementa captura do quadro recente, métricas separadas e preview opcional. Esses ajustes ainda precisam da webcam para validação.
3. Para melhorar qualidade: investigar a perda da quantização em um conjunto representativo, com comparação de saídas e por camada; testar calibração estratificada de fogo/fumaça/negativos e dimensões dos alvos, sempre separada da validação. Saída inteira tem escala única próxima de 0,01 nas variantes atuais, compartilhando coordenadas e escores; avaliar cabeça/saída em float é um experimento possível, não uma correção garantida nem já realizada.
4. Se a quantização pós-treino continuar degradando qualidade, avaliar QAT com uma cadeia de exportação compatível. A documentação do TensorFlow apresenta QAT como alternativa para melhorar acurácia da quantização; isso exige implementação e avaliação, não apenas um parâmetro nesta célula: https://www.tensorflow.org/model_optimization/guide/quantization/training .
5. Para ganhar margem real para 10 FPS, a próxima arquitetura a experimentar é YOLOv5n, treinada/refinada com o dataset e comparada ao modelo atual. A variante v7.0 é menor que YOLOv5s; isso não garante velocidade nem qualidade na placa: https://github.com/ultralytics/yolov5/tree/v7.0#pretrained-checkpoints . Reduzir a rede pode permitir manter resolução maior, importante para fumaça e focos pequenos. Ainda não foi treinada nesta rodada.
6. Antes de encerrar: avaliação no teste separado, cenas negativas, fumaça discreta, focos pequenos, variação de luz, estabilidade prolongada e atraso até a detecção com a webcam. Definir recall/precisão e taxa de falsos alertas aceitáveis junto da meta de FPS. Não foi implementado nem validado sistema de alarme.

## Reprodução e arquivos

`comparacao.csv` reúne métricas; `validacao/` contém resultados por modelo e por limiar, amostra e calibração; `raspberry/` contém benchmarks, CSVs e tempos. `videos/` reúne as comparações históricas a 2 FPS e duas simulações a 6,2 FPS, com parâmetros 0,15/0,15 e 0,17/0,17. O índice em `videos/README.md` apresenta os arquivos por configuração. As simulações mantêm o tempo da cena, mas não demonstram FPS real de captura USB.

O script USB atualizado está na placa em `~/incendio-edge/scripts/webcam_usb.py` e copiado em `../scripts/raspberry/`. Sintaxe, ajuda e pós-processamento com caixas sintéticas foram verificados. A webcam não estava disponível: captura USB e preview ainda não passaram por teste físico. A idade de quadro registrada pelo script começa quando `cap.read()` retorna, excluindo atraso desconhecido do sensor/driver.

A organização preservou o treinamento original completo no ZIP de `treinamento/`, conferido contra os 28 arquivos extraídos por SHA256. Modelos finais também foram conferidos contra a validação. As cópias redundantes, INT8 inicial defeituoso, prévias repetidas e intermediários foram removidos após conferência do inventário. Os inventários de organização permanecem locais; os hashes dos modelos estão versionados em `manifests/modelos.json`.
