# Revisar e preparar os negativos

As coletas devem estar encerradas. Os quadros e predições permanecem locais em `dados/coletas/`; nada é aprovado ou rotulado automaticamente. A revisão ocorre em uma galeria HTML offline, sem envio de imagens a serviços externos.

## Preparar a revisão

Na raiz do projeto, usando o ambiente Python com OpenCV do Raspberry:

```bash
.venv/bin/python scripts/revisao/preparar_revisao.py --root dados/coletas --output dados/coletas/revisao/lote-01
```

O programa valida os hashes e a leitura de cada imagem, identifica sessão/modelo e gera:

- `index.html`: galeria interativa, com caixas opcionais e decisões por imagem.
- `manifest.json`: procedência, hashes e estados inicialmente pendentes.
- `folhas/`: folhas de contato numeradas para inspeção rápida.

Para revisar no PC, copiar as sessões e a pasta de revisão preservando a estrutura de `dados/coletas/`, e abrir `index.html` no navegador. As imagens usam caminhos relativos; a galeria não precisa de servidor. Não colocar a coleta no Git.

## Decidir

1. Aprovar como negativo somente o quadro que não contenha fogo nem fumaça. O botão da página aprova apenas os itens ainda pendentes; não desfaz descartes anteriores.
2. Descartar casos duvidosos, corrompidos e repetições que não acrescentem diversidade. Uma indicação de semelhança é apenas sugestão, não motivo automático para exclusão. Reflexos e iluminação forte são importantes exemplos negativos e não devem ser descartados só por isso.
3. Usar a observação para registrar sombra, reflexo, saturação, repetição ou outro motivo.
4. Confirmar a revisão dos aprovados e baixar `revisao-falsos-positivos.json`. O progresso pode ficar no navegador, mas somente o arquivo baixado permite exportar o conjunto.

## Exportar depois da revisão

Copiar o JSON de decisões para junto da revisão e executar:

```bash
.venv/bin/python scripts/revisao/exportar_negativos.py --root dados/coletas --manifest dados/coletas/revisao/lote-01/manifest.json --decisions dados/coletas/revisao/lote-01/revisao-falsos-positivos.json --output dados/negativos/lote-01 --split train
```

Somente `approved_negative` entra na saída. Cada imagem aprovada recebe um arquivo YOLO vazio; nenhuma caixa prevista vira rótulo positivo. Pendentes e descartados são excluídos. O programa exige confirmação da revisão, lote e hashes correspondentes; rejeita IDs repetidos, arquivos modificados e destinos já preenchidos. Duplicatas com bytes idênticos são exportadas só uma vez, com registro de procedência.

`dados/negativos/` também é local e ignorado pelo Git. A saída é um pacote complementar, não um dataset completo para treinar apenas com negativos. O próximo passo é misturar os aprovados com os dados originais e positivos, preservando o conjunto original de validação e obtendo sessões independentes para testar os falsos alertas. Não sortear quadros vizinhos desta coleta entre treino e teste.

Para validar o fluxo com imagens sintéticas:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
