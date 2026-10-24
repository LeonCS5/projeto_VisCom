"""Reconhecimento facial (Haar Cascade + LBPH, ambos do OpenCV) e chamada automática.

Estrutura esperada do banco de rostos:
    faces/<nome_do_aluno>/<foto>.jpg   (várias fotos por aluno)
"""
import csv
import os
from datetime import datetime

import cv2
import numpy as np

PASTA_FACES = "faces"
MODELO = "modelo_lbph.yml"
ROTULOS = "rotulos.txt"
TAM_FACE = (200, 200)
# Quanto MENOR a confiança do LBPH, melhor o casamento. Acima disso = "desconhecido".
LIMIAR_CONFIANCA = 80.0

_detector = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


def _criar_reconhecedor():
    if not hasattr(cv2, "face"):
        raise RuntimeError("Módulo cv2.face ausente. Instale: pip install opencv-contrib-python")
    return cv2.face.LBPHFaceRecognizer_create()


def detectar_faces(img, escala=1.2, vizinhos=5):
    """Retorna lista de (x, y, w, h)."""
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    return list(_detector.detectMultiScale(g, escala, vizinhos, minSize=(60, 60)))


def _recorte_face(img, caixa):
    x, y, w, h = caixa
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(cv2.equalizeHist(g[y:y + h, x:x + w]), TAM_FACE)


def desenhar_deteccoes(img):
    saida = img.copy()
    caixas = detectar_faces(img)
    for x, y, w, h in caixas:
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return saida, len(caixas)


def treinar(pasta=PASTA_FACES):
    """Treina o LBPH com as fotos em faces/<nome>/ e salva modelo + rótulos."""
    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"Pasta '{pasta}' não existe. Cadastre alunos primeiro.")
    nomes = sorted(d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d)))
    amostras, ids = [], []
    for i, nome in enumerate(nomes):
        for arq in os.listdir(os.path.join(pasta, nome)):
            img = cv2.imread(os.path.join(pasta, nome, arq))
            if img is None:
                continue
            caixas = detectar_faces(img)
            if len(caixas) == 0:
                continue
            maior = max(caixas, key=lambda c: c[2] * c[3])
            amostras.append(_recorte_face(img, maior))
            ids.append(i)
    if not amostras:
        raise RuntimeError("Nenhum rosto encontrado nas fotos de treinamento.")
    rec = _criar_reconhecedor()
    rec.train(amostras, np.array(ids))
    rec.write(MODELO)
    with open(ROTULOS, "w", encoding="utf-8") as f:
        f.write("\n".join(nomes))
    return nomes, len(amostras)


def _carregar():
    if not (os.path.exists(MODELO) and os.path.exists(ROTULOS)):
        raise FileNotFoundError("Modelo não treinado. Use a opção de treinar primeiro.")
    rec = _criar_reconhecedor()
    rec.read(MODELO)
    with open(ROTULOS, encoding="utf-8") as f:
        nomes = f.read().split("\n")
    return rec, nomes


def reconhecer(img, rec=None, nomes=None):
    """Retorna (imagem anotada, [nomes reconhecidos])."""
    if rec is None:
        rec, nomes = _carregar()
    saida = img.copy()
    achados = []
    for caixa in detectar_faces(img):
        idx, conf = rec.predict(_recorte_face(img, caixa))
        nome = nomes[idx] if conf <= LIMIAR_CONFIANCA else "desconhecido"
        if nome != "desconhecido":
            achados.append(nome)
        x, y, w, h = caixa
        cor = (0, 255, 0) if nome != "desconhecido" else (0, 0, 255)
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(saida, f"{nome} ({conf:.0f})", (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return saida, achados


def cadastrar_pela_webcam(nome, quantidade=15, camera=0):
    """Captura `quantidade` fotos com rosto detectado (tecle ESPAÇO a cada captura, ESC sai)."""
    pasta = os.path.join(PASTA_FACES, nome)
    os.makedirs(pasta, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")
    salvas = len(os.listdir(pasta))
    meta = salvas + quantidade
    try:
        while salvas < meta:
            ok, quadro = cap.read()
            if not ok:
                break
            vis, n = desenhar_deteccoes(quadro)
            cv2.putText(vis, f"{nome}: {salvas}/{meta} - ESPACO captura, ESC sai",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Cadastro", vis)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            if tecla == 32 and n == 1:
                cv2.imwrite(os.path.join(pasta, f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"), quadro)
                salvas += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return salvas


def _registrar_presenca(presentes, arquivo):
    """Anexa presentes ao CSV (data, hora, nome); ignora quem já consta no dia."""
    hoje = f"{datetime.now():%Y-%m-%d}"
    ja = set()
    if os.path.exists(arquivo):
        with open(arquivo, newline="", encoding="utf-8") as f:
            ja = {l[2] for l in csv.reader(f) if len(l) >= 3 and l[0] == hoje}
    novos = [n for n in sorted(presentes) if n not in ja]
    novo_arquivo = not os.path.exists(arquivo)
    with open(arquivo, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo_arquivo:
            w.writerow(["data", "hora", "aluno"])
        for n in novos:
            w.writerow([hoje, f"{datetime.now():%H:%M:%S}", n])
    return novos


def chamada_webcam(arquivo="chamada.csv", camera=0, confirmacoes=5):
    """Chamada automática: um aluno é marcado presente após `confirmacoes` quadros
    reconhecidos (reduz falsos positivos). ESC encerra e grava o CSV.
    Retorna (presentes, ausentes)."""
    rec, nomes = _carregar()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")
    contagem, presentes = {}, set()
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            vis, achados = reconhecer(quadro, rec, nomes)
            for n in set(achados):
                contagem[n] = contagem.get(n, 0) + 1
                if contagem[n] >= confirmacoes:
                    presentes.add(n)
            cv2.putText(vis, f"Presentes: {len(presentes)}/{len(nomes)} - ESC encerra",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Chamada", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    _registrar_presenca(presentes, arquivo)
    return sorted(presentes), sorted(set(nomes) - presentes)
