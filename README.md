# projeto_VisCom

Software de processamento de imagens e vídeo com menu em terminal (Python + OpenCV).

## Executar

```
pip install -r requirements.txt
python main.py
```

## Módulos

- `main.py` – menu e interação com o usuário
- `processamento.py` – operações básicas, filtros, histograma, binarização, quantização, crescimento de regiões, morfologia
- `reconhecimento.py` – detecção (Haar Cascade), reconhecimento (LBPH) e chamada automática

## Reconhecimento facial / chamada

1. Cadastre cada aluno pela webcam (menu) ou coloque fotos em `faces/<nome_do_aluno>/*.jpg`.
2. Use "Treinar modelo" (gera `modelo_lbph.yml` e `rotulos.txt`).
3. Use "Chamada automática": o aluno é marcado presente após ser reconhecido em vários quadros seguidos; ao teclar ESC o resultado é gravado em `chamada.csv` (data, hora, aluno).

O limiar de decisão (`LIMIAR_CONFIANCA` em `reconhecimento.py`) pode precisar de ajuste conforme iluminação e câmera.
