"""Funções de processamento de imagens (todas recebem/retornam imagens BGR do OpenCV,
exceto quando indicado). Nenhuma altera a imagem de entrada."""
from collections import deque

import cv2
import numpy as np


def _bgr(img):
    """Garante 3 canais (útil para exibir/combinar resultados em cinza)."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


# ---------- Operações básicas ----------

def inverter(img):
    return cv2.bitwise_not(img)


def brilho_contraste(img, contraste=1.0, brilho=0):
    """g(x) = contraste * f(x) + brilho, com saturação em [0, 255]."""
    return cv2.convertScaleAbs(img, alpha=contraste, beta=brilho)


def somar(img, outra, peso=0.5):
    """Soma ponderada (mistura) com a segunda imagem redimensionada."""
    outra = cv2.resize(_bgr(outra), (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1 - peso, outra, peso, 0)


def subtrair(img, outra):
    """Diferença absoluta entre as imagens."""
    outra = cv2.resize(_bgr(outra), (img.shape[1], img.shape[0]))
    return cv2.absdiff(img, outra)


def cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def para_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv_para_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def redimensionar(img, largura=None, altura=None, escala=None):
    h, w = img.shape[:2]
    if escala:
        largura, altura = int(w * escala), int(h * escala)
    elif largura and not altura:
        altura = int(h * largura / w)
    elif altura and not largura:
        largura = int(w * altura / h)
    interp = cv2.INTER_AREA if largura < w else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, largura), max(1, altura)), interpolation=interp)


def recortar(img, x, y, w, h):
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        raise ValueError("ROI vazia ou fora da imagem.")
    return img[y:y + h, x:x + w].copy()


def selecionar_roi(img):
    """Seleção interativa com o mouse. Retorna (x, y, w, h)."""
    x, y, w, h = cv2.selectROI("Selecione a ROI e tecle ENTER (C cancela)", img, False)
    cv2.destroyAllWindows()
    return int(x), int(y), int(w), int(h)


# ---------- Filtragem e realce ----------

def _impar(k):
    k = max(1, int(k))
    return k if k % 2 else k + 1


def filtro_media(img, k=5):
    return cv2.blur(img, (_impar(k), _impar(k)))


def filtro_mediana(img, k=5):
    return cv2.medianBlur(img, _impar(k))


def filtro_gaussiano(img, k=5, sigma=0):
    return cv2.GaussianBlur(img, (_impar(k), _impar(k)), sigma)


def sobel(img, k=3):
    g = img if img.ndim == 2 else cinza(img)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=_impar(k))
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=_impar(k))
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def canny(img, t1=100, t2=200):
    g = img if img.ndim == 2 else cinza(img)
    return cv2.Canny(g, t1, t2)


# ---------- Histograma e intensidade ----------

def calcular_histogramas(img):
    """Retorna dict {rótulo: histograma de 256 bins}."""
    if img.ndim == 2:
        return {"cinza": cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()}
    return {nome: cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
            for i, nome in enumerate(("azul", "verde", "vermelho"))}


def mostrar_histograma(img, titulo="Histograma"):
    import matplotlib.pyplot as plt
    cores = {"cinza": "k", "azul": "b", "verde": "g", "vermelho": "r"}
    plt.figure(titulo)
    for nome, h in calcular_histogramas(img).items():
        plt.plot(h, color=cores[nome], label=nome)
    plt.xlim(0, 255)
    plt.xlabel("Intensidade")
    plt.ylabel("Pixels")
    plt.title(titulo)
    plt.legend()
    plt.show()


def equalizar(img):
    """Equaliza o histograma (em cinza, ou só o canal V para imagens coloridas)."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    hsv = para_hsv(img)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return hsv_para_bgr(hsv)


def separar_canais(img):
    """Retorna (B, G, R) em cinza e uma montagem colorida de cada canal isolado."""
    b, g, r = cv2.split(_bgr(img))
    z = np.zeros_like(b)
    coloridos = [cv2.merge([b, z, z]), cv2.merge([z, g, z]), cv2.merge([z, z, r])]
    return (b, g, r), np.hstack(coloridos)


def estatisticas_canais(img):
    """Média e desvio-padrão por canal, como texto."""
    linhas = []
    for nome, c in zip("BGR", cv2.split(_bgr(img))):
        linhas.append(f"  {nome}: média={c.mean():6.2f}  desvio={c.std():6.2f}  "
                      f"mín={c.min()}  máx={c.max()}")
    return "\n".join(linhas)


def binarizar(img, metodo="otsu", limiar=127, bloco=11, c=2):
    g = img if img.ndim == 2 else cinza(img)
    if metodo == "fixo":
        return cv2.threshold(g, limiar, 255, cv2.THRESH_BINARY)[1]
    if metodo == "adaptativo":
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, _impar(max(3, bloco)), c)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def quantizar(img, niveis=4):
    """Reduz cada canal para `niveis` valores distintos."""
    niveis = int(np.clip(niveis, 2, 256))
    passo = 256 / niveis
    q = (np.floor(img / passo) * (255 / (niveis - 1))).astype(np.uint8)
    return q


# ---------- Segmentação e morfologia ----------

def crescimento_regioes(img, semente, tolerancia=20):
    """Crescimento de regiões a partir de uma semente (x, y), vizinhança-8.
    Um pixel é agregado se sua distância à cor da semente for <= tolerância.
    Retorna a máscara binária (0/255)."""
    g = img.astype(np.int32)
    if g.ndim == 2:
        g = g[:, :, None]
    h, w = g.shape[:2]
    sx, sy = semente
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("Semente fora da imagem.")
    ref = g[sy, sx]
    visitado = np.zeros((h, w), np.uint8)
    visitado[sy, sx] = 255
    fila = deque([(sx, sy)])
    while fila:
        x, y = fila.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visitado[ny, nx]:
                    if np.abs(g[ny, nx] - ref).max() <= tolerancia:
                        visitado[ny, nx] = 255
                        fila.append((nx, ny))
    return visitado


def selecionar_semente(img):
    """Clique com o botão esquerdo para escolher a semente (ESC cancela)."""
    ponto = []
    nome = "Clique na semente"

    def cb(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            ponto.append((x, y))

    cv2.imshow(nome, img)
    cv2.setMouseCallback(nome, cb)
    while not ponto:
        if cv2.waitKey(30) == 27 or cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return ponto[0] if ponto else None


def destacar_regiao(img, mascara):
    """Pinta a região segmentada de verde sobre a imagem."""
    saida = _bgr(img).copy()
    cor = np.zeros_like(saida)
    cor[:] = (0, 255, 0)
    saida[mascara > 0] = cv2.addWeighted(saida, 0.4, cor, 0.6, 0)[mascara > 0]
    return saida


_OPS_MORF = {
    "erosao": cv2.MORPH_ERODE,
    "dilatacao": cv2.MORPH_DILATE,
    "abertura": cv2.MORPH_OPEN,
    "fechamento": cv2.MORPH_CLOSE,
}


def morfologia(img, operacao="erosao", k=3, formato="retangulo", iteracoes=1):
    forma = {"retangulo": cv2.MORPH_RECT, "elipse": cv2.MORPH_ELLIPSE,
             "cruz": cv2.MORPH_CROSS}[formato]
    kernel = cv2.getStructuringElement(forma, (_impar(k), _impar(k)))
    return cv2.morphologyEx(img, _OPS_MORF[operacao], kernel, iterations=int(iteracoes))


# ---------- Ajustes extras e detecção de formas ----------

def saturacao(img, fator=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * fator, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def nitidez(img, quantidade=1.0, raio=2):
    """Máscara de nitidez (unsharp mask): realça o que o desfoque gaussiano remove."""
    k = _impar(raio * 2 + 1)
    desfoque = cv2.GaussianBlur(img, (k, k), 0)
    return cv2.addWeighted(img, 1 + quantidade, desfoque, -quantidade, 0)


def rotacionar(img, angulo, expandir=True):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    if not expandir:
        return cv2.warpAffine(img, m, (w, h))
    cos, sen = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sen + w * cos), int(h * cos + w * sen)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh))


def espelhar(img, modo="horizontal"):
    return cv2.flip(img, {"horizontal": 1, "vertical": 0, "ambos": -1}[modo])


def contornos(img, t1=50, t2=150, area_min=100):
    """Contornos externos a partir das bordas de Canny. Retorna (imagem desenhada, quantidade)."""
    bordas = cv2.dilate(canny(cv2.GaussianBlur(_bgr(img), (5, 5), 0), t1, t2), np.ones((3, 3), np.uint8))
    achados, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    achados = [c for c in achados if cv2.contourArea(c) >= area_min]
    saida = _bgr(img).copy()
    cv2.drawContours(saida, achados, -1, (0, 255, 0), 2)
    return saida, len(achados)


def linhas(img, limiar=80, comprimento_min=40, falha_max=10):
    """Segmentos de reta (Hough probabilístico). Retorna (imagem desenhada, quantidade)."""
    bordas = canny(img, 50, 150)
    achadas = cv2.HoughLinesP(bordas, 1, np.pi / 180, limiar, minLineLength=comprimento_min,
                              maxLineGap=falha_max)
    saida = _bgr(img).copy()
    n = 0 if achadas is None else len(achadas)
    for x1, y1, x2, y2 in ([] if achadas is None else achadas[:, 0]):
        cv2.line(saida, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return saida, n


def circulos(img, sensibilidade=30, raio_min=10, raio_max=0):
    """Círculos (Hough). `sensibilidade` baixa = mais círculos (e mais falsos). raio_max 0 = sem limite."""
    g = cv2.medianBlur(img if img.ndim == 2 else cinza(img), 5)
    achados = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, max(10, g.shape[0] // 8),
                               param1=100, param2=max(5, sensibilidade),
                               minRadius=int(raio_min), maxRadius=int(raio_max))
    saida = _bgr(img).copy()
    n = 0 if achados is None else achados.shape[1]
    for x, y, r in ([] if achados is None else np.round(achados[0]).astype(int)):
        cv2.circle(saida, (x, y), r, (255, 0, 255), 2)
        cv2.circle(saida, (x, y), 2, (0, 255, 255), 3)
    return saida, n


def cantos(img, maximo=100, qualidade=0.01):
    """Cantos de Shi-Tomasi. Retorna (imagem desenhada, quantidade)."""
    g = img if img.ndim == 2 else cinza(img)
    achados = cv2.goodFeaturesToTrack(g, int(maximo), max(1e-4, qualidade), 10)
    saida = _bgr(img).copy()
    n = 0 if achados is None else len(achados)
    for x, y in ([] if achados is None else np.round(achados[:, 0]).astype(int)):
        cv2.circle(saida, (x, y), 4, (0, 200, 255), -1)
    return saida, n
