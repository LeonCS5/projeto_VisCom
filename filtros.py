"""Filtros prontos no estilo de apps de fotos. Cada filtro recebe e devolve uma imagem BGR
do mesmo tamanho; são compostos só de operações do módulo de processamento/OpenCV."""
import cv2
import numpy as np

import processamento as p


# ---------- blocos de construção ----------

def _contraste(img, c, brilho=0):
    return cv2.convertScaleAbs(img, alpha=c, beta=brilho + 128 * (1 - c))


def _saturacao(img, f):
    return p.saturacao(img, f)


def _ganhos(img, b=1.0, g=1.0, r=1.0):
    """Multiplica cada canal (tinge a foto: mais r = mais quente, mais b = mais fria)."""
    return np.clip(img.astype(np.float32) * np.array([b, g, r], np.float32), 0, 255).astype(np.uint8)


def _desbotar(img, k):
    """Levanta os pretos, dando o aspecto 'lavado' de foto antiga."""
    lift = 255 * k
    return np.clip(img.astype(np.float32) * (1 - k) + lift, 0, 255).astype(np.uint8)


def _vinheta(img, forca):
    h, w = img.shape[:2]
    gy = cv2.getGaussianKernel(h, h * 0.55)
    gx = cv2.getGaussianKernel(w, w * 0.55)
    m = gy @ gx.T
    m = m / m.max()
    return np.clip(img * (1 - forca + forca * m[:, :, None]), 0, 255).astype(np.uint8)


def _granulado(img, forca, semente=0):
    ruido = np.random.default_rng(semente).normal(0, forca, img.shape[:2])
    return np.clip(img.astype(np.float32) + ruido[:, :, None], 0, 255).astype(np.uint8)


def _cinza3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _sepia(img):
    m = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]])
    return cv2.transform(img, m)  # linhas = saída B, G, R; colunas = entrada B, G, R


def _escala(img):
    """Tamanho de referência para parâmetros espaciais (filtros ficam iguais em miniatura)."""
    return min(img.shape[:2])


# ---------- filtros ----------

def original(img):
    return img.copy()


def clarendon(img):
    return _ganhos(_saturacao(_contraste(img, 1.2, 8), 1.3), 1.06, 1.0, 0.96)


def gingham(img):
    return _ganhos(_desbotar(_saturacao(img, 0.8), 0.1), 1.0, 1.0, 1.03)


def juno(img):
    return _ganhos(_saturacao(_contraste(img, 1.1), 1.3), 0.93, 1.0, 1.08)


def lark(img):
    return _ganhos(_saturacao(_contraste(img, 1.05, 14), 0.9), 1.0, 1.04, 1.02)


def valencia(img):
    return _ganhos(_desbotar(_contraste(img, 1.05), 0.08), 0.9, 1.0, 1.1)


def nashville(img):
    return _ganhos(_desbotar(_contraste(img, 1.1), 0.12), 1.0, 0.96, 1.12)


def lofi(img):
    return _vinheta(_saturacao(_contraste(img, 1.35), 1.5), 0.4)


def moon(img):
    return _contraste(_cinza3(img), 1.15, 6)


def noir(img):
    return _vinheta(_contraste(_cinza3(img), 1.6), 0.55)


def sepia(img):
    return _sepia(img)


def vintage(img):
    base = cv2.addWeighted(img, 0.4, _sepia(img), 0.6, 0)
    return _vinheta(_granulado(_desbotar(base, 0.12), 8), 0.35)


def frio(img):
    return _ganhos(img, 1.14, 1.0, 0.9)


def quente(img):
    return _ganhos(img, 0.88, 1.0, 1.12)


def pele_suave(img):
    s = _escala(img)
    d = max(5, int(s * 0.015)) | 1
    out = img
    for _ in range(2):
        out = cv2.bilateralFilter(out, d, 55, d * 2)
    return cv2.addWeighted(out, 0.9, img, 0.1, 6)


def desenho(img):
    s = _escala(img)
    suave = img
    for _ in range(2):
        suave = cv2.bilateralFilter(suave, max(5, int(s * 0.015)) | 1, 60, 12)
    suave = p.quantizar(suave, 8)
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    bordas = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
    return cv2.bitwise_and(suave, cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR))


def lapis(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(3, int(_escala(img) * 0.04)) | 1
    desfoque = cv2.GaussianBlur(255 - g, (k, k), 0)
    esboco = cv2.divide(g, 255 - desfoque, scale=256)  # color dodge
    return cv2.cvtColor(esboco, cv2.COLOR_GRAY2BGR)


def poster(img):
    return _saturacao(p.quantizar(img, 4), 1.25)


def pixelado(img):
    h, w = img.shape[:2]
    bloco = max(2, _escala(img) // 40)
    pequeno = cv2.resize(img, (max(1, w // bloco), max(1, h // bloco)), interpolation=cv2.INTER_AREA)
    return cv2.resize(pequeno, (w, h), interpolation=cv2.INTER_NEAREST)


def vinheta(img):
    return _vinheta(img, 0.65)


# nome -> (função, descrição curta)
FILTROS = {
    "Original": (original, "Sem filtro"),
    "Clarendon": (clarendon, "Mais contraste e cores vivas, tom levemente frio"),
    "Gingham": (gingham, "Suave e desbotado, clima delicado"),
    "Juno": (juno, "Cores quentes e saturadas"),
    "Lark": (lark, "Claro e arejado, bom para paisagens"),
    "Valencia": (valencia, "Tom quente e leve efeito antigo"),
    "Nashville": (nashville, "Rosado e nostálgico"),
    "Lo-fi": (lofi, "Cores fortes, contraste alto e cantos escuros"),
    "Moon": (moon, "Preto e branco suave"),
    "Noir": (noir, "Preto e branco dramático com vinheta"),
    "Sépia": (sepia, "Tom marrom de foto antiga"),
    "Vintage": (vintage, "Sépia + grão + cantos escuros"),
    "Frio": (frio, "Tom azulado"),
    "Quente": (quente, "Tom alaranjado"),
    "Pele suave": (pele_suave, "Suaviza a textura preservando contornos"),
    "Desenho": (desenho, "Efeito cartoon com contornos"),
    "Lápis": (lapis, "Esboço a lápis"),
    "Pôster": (poster, "Poucas cores, estilo cartaz"),
    "Pixelado": (pixelado, "Blocos grandes de pixels"),
    "Vinheta": (vinheta, "Escurece os cantos"),
}


def aplicar_filtro(nome, img, intensidade=1.0):
    """Aplica o filtro `nome`; `intensidade` (0..1) mistura o resultado com a original."""
    out = FILTROS[nome][0](img)
    if intensidade >= 1.0:
        return out
    return cv2.addWeighted(img, 1 - intensidade, out, intensidade, 0)
