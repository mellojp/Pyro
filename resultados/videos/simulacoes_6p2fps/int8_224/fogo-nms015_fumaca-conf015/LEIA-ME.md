# Prévia da cadência de câmera: INT8 224, 6,2 FPS

Três vídeos com inferência real em quadros distintos amostrados no tempo original. A reprodução tem 6,2 quadros por segundo, sem interpolação nem aceleração da cena. A diferença de duração em relação ao original é limitada ao arredondamento de um quadro de saída.

Configuração: modelo INT8 224, duas threads, confiança de fogo 0,25, confiança de fumaça 0,15, NMS IoU fogo 0,15 e fumaça 0,45; filtro de contenção desativado. Aqui 0,15 para fumaça significa confiança, conforme o teste anterior, e não IoU do NMS.

É uma simulação visual de cadência fixa baseada nos 6,2 FPS medidos anteriormente. Não reproduz variações de latência, atraso do sensor/USB, JPEG ou transmissão de preview de uma webcam. O tempo necessário para gerar os arquivos offline não determina sua taxa de reprodução. As caixas correspondem ao próprio quadro mostrado. Sem áudio.

Os arquivos `video*-metricas.json` registram os índices dos quadros de origem, os tempos de inferência e a configuração. Não interpretar a média da geração offline como benchmark de captura USB, pois há leitura e gravação intercaladas. O script de reprodução está em `scripts/raspberry/simular_camera_video.py`, relativo à raiz do projeto.
