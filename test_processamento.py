import os

import cv2
import numpy as np
import pytest

import processamento as p
import reconhecimento as rf


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    im = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
    return im


@pytest.fixture
def quadrado():
    im = np.zeros((100, 100, 3), np.uint8)
    im[30:70, 30:70] = 200
    return im


def test_inverter(img):
    assert np.array_equal(p.inverter(p.inverter(img)), img)


def test_nao_altera_entrada(img):
    copia = img.copy()
    for f in (p.inverter, p.cinza, p.para_hsv, p.equalizar, p.filtro_media,
              p.filtro_mediana, p.filtro_gaussiano, p.sobel, p.canny, p.quantizar):
        f(img)
    assert np.array_equal(img, copia)


def test_brilho_contraste_satura(img):
    out = p.brilho_contraste(img, 2.0, 100)
    assert out.dtype == np.uint8 and out.max() == 255


def test_somar_subtrair_tamanhos_diferentes(img):
    outra = np.full((40, 40, 3), 50, np.uint8)
    assert p.somar(img, outra).shape == img.shape
    assert p.subtrair(img, img).max() == 0
    assert p.subtrair(img, outra.astype(np.uint8)).shape == img.shape


def test_cinza_hsv(img):
    assert p.cinza(img).shape == img.shape[:2]
    assert p.para_hsv(img).shape == img.shape
    assert p.hsv_para_bgr(p.para_hsv(img)).shape == img.shape


def test_redimensionar(img):
    assert p.redimensionar(img, escala=0.5).shape[:2] == (40, 60)
    assert p.redimensionar(img, 60, 20).shape[:2] == (20, 60)
    assert p.redimensionar(img, largura=60).shape[:2] == (40, 60)


def test_recortar(img):
    assert p.recortar(img, 10, 10, 20, 30).shape[:2] == (30, 20)
    assert p.recortar(img, 100, 70, 50, 50).shape[:2] == (10, 20)  # limitado à borda
    with pytest.raises(ValueError):
        p.recortar(img, 500, 500, 10, 10)


def test_filtros_kernel_par_vira_impar(img):
    for f in (p.filtro_media, p.filtro_mediana, p.filtro_gaussiano):
        assert f(img, 4).shape == img.shape


def test_bordas(quadrado):
    assert p.sobel(quadrado).max() > 0
    assert p.canny(quadrado).max() == 255


def test_histograma(img):
    h = p.calcular_histogramas(img)
    assert set(h) == {"azul", "verde", "vermelho"}
    assert h["azul"].sum() == img.shape[0] * img.shape[1]
    assert set(p.calcular_histogramas(p.cinza(img))) == {"cinza"}


def test_equalizar(img):
    assert p.equalizar(img).shape == img.shape
    assert p.equalizar(p.cinza(img)).ndim == 2


def test_canais(img):
    (b, g, r), montagem = p.separar_canais(img)
    assert montagem.shape == (80, 360, 3)
    assert np.array_equal(b, img[:, :, 0])
    assert "B:" in p.estatisticas_canais(img)


@pytest.mark.parametrize("metodo", ["fixo", "adaptativo", "otsu"])
def test_binarizar(img, metodo):
    assert set(np.unique(p.binarizar(img, metodo))) <= {0, 255}


def test_quantizar(img):
    for n in (2, 4, 8):
        assert len(np.unique(p.quantizar(img, n))) <= n


def test_crescimento_regioes(quadrado):
    m = p.crescimento_regioes(quadrado, (50, 50), 10)
    assert (m > 0).sum() == 40 * 40
    fundo = p.crescimento_regioes(quadrado, (5, 5), 10)
    assert (fundo > 0).sum() == 100 * 100 - 40 * 40
    assert p.crescimento_regioes(p.cinza(quadrado), (50, 50), 10).shape == (100, 100)
    with pytest.raises(ValueError):
        p.crescimento_regioes(quadrado, (500, 5), 10)
    assert p.destacar_regiao(quadrado, m).shape == quadrado.shape


def test_morfologia(quadrado):
    b = p.binarizar(quadrado, "otsu")
    area = lambda x: int((x > 0).sum())
    assert area(p.morfologia(b, "erosao", 3)) < area(b)
    assert area(p.morfologia(b, "dilatacao", 3)) > area(b)
    assert area(p.morfologia(b, "abertura", 3)) <= area(b)
    assert area(p.morfologia(b, "fechamento", 3)) >= area(b)


def test_detectar_sem_rosto(quadrado):
    saida, n = rf.desenhar_deteccoes(quadrado)
    assert n == 0 and saida.shape == quadrado.shape


def test_registrar_presenca(tmp_path):
    arq = str(tmp_path / "c.csv")
    assert rf._registrar_presenca({"ana", "bia"}, arq) == ["ana", "bia"]
    assert rf._registrar_presenca({"ana", "caio"}, arq) == ["caio"]  # ana já consta hoje
    with open(arq, encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 4


def test_treinar_sem_pasta(tmp_path):
    with pytest.raises(FileNotFoundError):
        rf.treinar(str(tmp_path / "nao_existe"))


def test_lbph_treina_e_reconhece(tmp_path, monkeypatch):
    """Substitui o detector por um que devolve a imagem toda, para testar o pipeline LBPH."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    base = {n: rng.integers(0, 256, (120, 120), dtype=np.uint8) for n in ("ana", "bia")}
    for n, b in base.items():
        os.makedirs(f"faces/{n}")
        for i in range(3):
            ruido = np.clip(b.astype(int) + rng.integers(-5, 6, b.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(f"faces/{n}/{i}.png", cv2.cvtColor(ruido, cv2.COLOR_GRAY2BGR))
    monkeypatch.setattr(rf, "detectar_faces", lambda im, *a, **k: [(0, 0, im.shape[1], im.shape[0])])
    nomes, qtd = rf.treinar()
    assert nomes == ["ana", "bia"] and qtd == 6
    _, achados = rf.reconhecer(cv2.cvtColor(base["bia"], cv2.COLOR_GRAY2BGR))
    assert achados == ["bia"]


def test_saturacao_nitidez(img):
    assert p.saturacao(img, 0).std(axis=2).max() <= 1  # sem saturação = cinza
    assert p.saturacao(img, 1.5).shape == img.shape
    assert np.array_equal(p.nitidez(img, 0.0), img)
    assert p.nitidez(img, 1.0).shape == img.shape


def test_rotacionar_espelhar(img):
    assert p.rotacionar(img, 0).shape == img.shape
    assert p.rotacionar(img, 90).shape[:2] == (120, 80)
    assert p.rotacionar(img, 45, expandir=False).shape == img.shape
    assert np.array_equal(p.espelhar(p.espelhar(img, "horizontal"), "horizontal"), img)
    assert np.array_equal(p.espelhar(img, "ambos"), p.espelhar(p.espelhar(img, "vertical"), "horizontal"))


def test_deteccoes_de_formas():
    im = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(im, (100, 100), 50, (255, 255, 255), -1)
    out, n = p.contornos(im)
    assert out.shape == im.shape and n >= 1
    out, n = p.circulos(im, 20, 30, 80)
    assert n >= 1
    quadro = np.zeros((200, 200, 3), np.uint8)
    cv2.rectangle(quadro, (40, 40), (160, 160), (255, 255, 255), 3)
    assert p.linhas(quadro, 50, 40, 5)[1] >= 4
    assert p.cantos(quadro, 20, 0.05)[1] >= 4
    assert p.contornos(np.zeros((50, 50, 3), np.uint8))[1] == 0
    assert p.cantos(np.zeros((50, 50, 3), np.uint8))[1] == 0
