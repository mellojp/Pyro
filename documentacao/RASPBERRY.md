# Fogo e fumaca: Raspberry Pi 3 Model B

Projeto: `/home/admin/incendio-edge`. Ambiente `.venv` preservado. Dependencias em `scripts/raspberry/requirements-referencia.txt`. As ferramentas locais de conexão com o Colab e suas credenciais ficam fora deste repositório.

Perfil atual: INT8 224, duas threads, confianca fogo 0.25 e fumaca 0.15, NMS IoU fogo 0.15 e fumaca 0.45, contencao desativada. O JSON `scripts/raspberry/perfil_atual.json` registra os padroes; nao e carregado automaticamente pelo script. Para variar, use argumentos CLI. Valores 0.17 e 0.18 sao aceitos. IoU maior no NMS permite mais sobreposicao; confianca maior filtra escores mais baixos.

## Estrutura

- `dados/videos/`: tres videos originais.
- `modelos/`: FP16 320; INT8 320, 256, 224 e 192.
- `scripts/raspberry/`: webcam, runtime, pos-processamento, simulacao e benchmarks atuais.
- `scripts/colab/`: fontes de exportacao e avaliacao.
- `arquivo/`: historico e registros locais de organizacao, preservados fora do Git.
- `resultados/raspberry/`: medicoes e CSVs dos experimentos atuais.
- `resultados/validacao/`: qualidade e limiares avaliados no Colab.
- `resultados/videos/comparacoes_2fps/`: previas selecionadas a 2 FPS.
- `resultados/videos/simulacoes_6p2fps/int8_224/fogo-nms015_fumaca-conf015/`: tres previas a 6.2 FPS com fumaca 0.15.
- `resultados/camera/`: destino das futuras medicoes da webcam.
- `arquivo/historico_experimentos.zip`: historico deduplicado por SHA256, incluindo INT8 inicial defeituoso e scripts antigos; nao usar esses pesos na aplicacao.
- `arquivo/organizacao/`: inventario, mapa de preservacao e verificacao final. Cada registro aponta para o conteudo no ZIP ou nas pastas atuais. Caminhos citados dentro de documentos historicos podem ser antigos.

## Webcam USB

```bash
cd ~/incendio-edge
./scripts/raspberry/iniciar_camera.sh --seconds 120 --preview-port 8080
# Teste de limiares intermediarios, ainda sem validacao de qualidade:
./scripts/raspberry/iniciar_camera.sh --fire-iou 0.18 --smoke-conf 0.17 --seconds 120 --preview-port 8080
# Alternativa mais rapida:
./scripts/raspberry/iniciar_camera.sh --model modelos/int8_192.tflite --seconds 120 --preview-port 8080
```

No PC: `ssh -L 8080:127.0.0.1:8080 USUARIO@IP_DO_RASPBERRY`, depois abrir `http://localhost:8080`.

O programa processa o quadro mais recente sem formar fila e separa FPS de captura/deteccao. Metricas: `resultados/camera/webcam-usb-metricas.json`. Para preservar varias rodadas, usar `--metrics outro-caminho.json`. Sem `--preview-port`, evita desenho e JPEG do preview. Captura USB ainda nao testada fisicamente.

## Reproducao

```bash
cd ~/incendio-edge
.venv/bin/python scripts/raspberry/simular_camera_video.py
.venv/bin/python scripts/raspberry/estudo_raspberry.py --model modelos/int8_224.tflite --name int8_224_nova_rodada
.venv/bin/python scripts/raspberry/benchmark.py 2
```

A simulacao tem fumaca 0.15 e cadencia fixa de 6.2 FPS, preserva o tempo da cena e substitui arquivos da simulacao anterior; use `--output outra-pasta` para preservar rodadas. O estudo comparativo conserva seu protocolo historico com fumaca 0.20, independentemente do padrao da webcam. O benchmark simples e de FP16 e mede somente a rede.

Medicoes offline com desenho: INT8 224 cerca de 6.2 FPS; INT8 192 cerca de 8.4 FPS. Nao sao taxas medidas de captura USB. mAP50 nas mesmas 1000 imagens: FP16 320 73.9%; INT8 224 60.0%; INT8 192 54.7%. O 224 e o candidato equilibrado. Ganhar cerca de 2 FPS exige passar de aproximadamente 160 para 122 ms por quadro, principalmente na rede; mudar limiares quase nao altera esse custo.

## Sincronizacao com o GitHub

A pasta ~/incendio-edge usa a mesma estrutura e a mesma origin do repositorio neste PC. Para receber atualizacoes, executar a partir dessa pasta:

```bash
git status --short
git pull --ff-only
git rev-parse HEAD
```

O commit deve coincidir com origin/main. Ambiente .venv, arquivo/, logs e inventarios de organizacao sao locais e ignorados pelo Git; nao sao copiados entre computadores. Alteracoes locais em arquivos versionados devem ser revisadas antes de atualizar.
