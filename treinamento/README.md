# Treinamento preservado

`pesos/best.pt` é o melhor checkpoint do treinamento/refino realizado no Colab, extraído do backup original e conferido por SHA256:

```text
08ab8085de112f14caf43780e6b312fa79997bd596fceca93f4ecbb3021712e5
```

`metricas/` preserva `results.csv`, `opt.yaml`, `hyp.yaml`, curvas e matriz de confusão. Os caminhos `/content/...` nas configurações são os caminhos originais da sessão de treino, e não caminhos exigidos neste clone. O refino registrado usou 20 épocas e imagem de tamanho 416; as exportações de inferência têm suas próprias resoluções.

O ZIP completo `resultados_treino_v4.zip` continua preservado localmente, mas não é versionado por conter cópias do checkpoint, modelos e outros intermediários. O dataset completo também não faz parte do repositório. O treinamento original e a validação comparativa posterior têm protocolos diferentes; não comparar seus números diretamente.

O checkpoint usa classes do YOLOv5 e deve ser carregado no ambiente compatível documentado em `scripts/colab/`.
