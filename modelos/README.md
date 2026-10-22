# Modelos

Todos os modelos têm duas classes, na ordem `fire`, `smoke`. O pós-processamento (confiança, NMS e desenho) é externo ao TFLite, em `scripts/raspberry/`.

| Arquivo | Entrada | Tipo da entrada/saída | Uso |
|---|---|---|---|
| `fp16_320.tflite` | 320x320 | float32 / float32 | Referência com pesos FP16 |
| `int8_320.tflite` | 320x320 | uint8 / uint8 | Exportação INT8 corrigida |
| `int8_256.tflite` | 256x256 | uint8 / uint8 | Comparação de resolução |
| `int8_224.tflite` | 224x224 | uint8 / uint8 | Candidato equilibrado atual |
| `int8_192.tflite` | 192x192 | uint8 / uint8 | Alternativa de menor latência |

O nome INT8 descreve a quantização interna; a interface de entrada e saída é uint8. O runtime consulta escala e zero point do próprio modelo. Imagens passam por letterbox, BGR para RGB e tratamento de quantização. Não basta mudar o tamanho do tensor de uma exportação para reproduzir outra resolução.

Hashes SHA256 em [modelos.json](../resultados/manifests/modelos.json). Checkpoint de origem em [treinamento/pesos/best.pt](../treinamento/pesos/best.pt). Os valores de AP e latência estão no [relatório](../resultados/RELATORIO.md); aparência satisfatória nos vídeos não substitui avaliação com rótulos.
