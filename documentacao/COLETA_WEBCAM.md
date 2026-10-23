# Coleta de falsos positivos e teste da webcam

Uma sessão confirmadamente sem fogo **e sem fumaça reais** permite coletar detecções incorretas como candidatos a negativos difíceis. A ausência dos dois objetos deve ser confirmada pelo operador. Se houver isqueiro, fogo, fumaça, vídeo em uma tela ou dúvida sobre o conteúdo, interromper a coleta e revisar os quadros; não aproveitar imagens que contenham objetos reais como negativos vazios.

## O que é salvo

- Quadro original da câmera, sem caixas nem texto, em `images/`, JPEG qualidade 95.
- `candidates.jsonl`: predições, escores, coordenadas na imagem original, sequência, horário UTC, dimensões e hash da imagem.
- `session.json`: parâmetros, modelo/hash, câmera, confirmação do operador, contadores e estado da sessão.

As predições **não são rótulos de treinamento**. Nenhum arquivo de rótulo YOLO é criado automaticamente. Todos os quadros começam como `pending`, para revisão humana.

Por padrão, salva no máximo um candidato a cada dois segundos, ignora imagens muito parecidas com a última salva e limita a sessão a 300 imagens. A comparação de semelhança é aproximada e pode pular alterações locais pequenas. Use várias cenas e condições de iluminação, em vez de acumular muitos quadros quase idênticos. Não são salvos quadros quando não há detecções acima dos limiares configurados.

## Executar FP16 em tempo real com coleta

A partir da raiz do repositório na placa, em uma pasta nova por sessão:

```bash
.venv/bin/python scripts/raspberry/webcam_usb.py --model modelos/fp16_320.tflite --threads 2 --fire-conf 0.32 --smoke-conf 0.25 --fire-iou 0.17 --preview-port 8081 --seconds 3600 --collect-dir dados/coletas/fp16-sessao-01 --confirm-no-fire-smoke --metrics resultados/camera/fp16-sessao-01.json
```

A opção de coleta exige `--confirm-no-fire-smoke`. Essa opção registra a confirmação do operador; não é uma verificação automática do conteúdo. Para testar só o FP16, omitir `--collect-dir` e `--confirm-no-fire-smoke`.

Os argumentos `--collect-interval`, `--collect-max-images` e `--collect-min-change` permitem ajustar a seleção. Um diretório já preenchido é rejeitado para evitar misturar sessões. Imagens de webcam, metadados de coleta e métricas de execução são locais e ignorados pelo Git.

O preview mostra o nome do modelo, as confianças, FPS e número de imagens coletadas. Para abrir pelo PC, com a porta local livre:

```bash
ssh -L 8081:127.0.0.1:8081 USUARIO@IP_DO_RASPBERRY
```

Abrir `http://localhost:8081`. Encerrar com Ctrl+C na sessão que executa o programa. Se iniciado em segundo plano, enviar SIGINT ao PID específico da execução para salvar as métricas finais.

## Depois da coleta

1. Revisar os originais e as predições; remover casos duvidosos, imagens corrompidas e repetições.
2. Aprovar como negativo somente o quadro em que não exista fogo nem fumaça. Quadros com objetos reais precisam ser anotados corretamente ou excluídos dessa coleta negativa.
3. Só então criar o conjunto de treino: imagem aprovada e rótulo YOLO vazio para o negativo. Não converter as caixas falsas em rótulos positivos.
4. Misturar os negativos com positivos e dados originais. Preservar sessões completas para validação/teste, sem dividir quadros vizinhos da mesma gravação entre os conjuntos.
5. Avaliar falsos positivos por tempo de vídeo e recall de exemplos reais; uma coleta apenas negativa não mede sensibilidade.

Um único modelo seleciona apenas os erros que ele próprio produz. É importante coletar erros do INT8 também e incluir ambientes variados, não apenas os candidatos do FP16.

## Comparação entre modelos

O FP16 disponível usa entrada 320x320; o INT8 mais usado usa 224x224. A comparação ao vivo muda resolução e quantização ao mesmo tempo, portanto não isola a causa das diferenças. Para avaliar a exportação, usar float e INT8 com mesma resolução nos mesmos quadros originais. O FP16 não é necessariamente mais rápido na CPU do Raspberry.

A escrita em disco da coleta entra no tempo total da iteração e pode afetar o FPS. Para benchmark de velocidade do modelo, fazer uma rodada separada sem coleta, com a mesma captura e configuração de preview.

## Verificação do coletor

No ambiente do Raspberry, a partir da raiz:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
