import numpy as np
import pytest

import filtros


@pytest.mark.parametrize("tamanho", [(64, 64), (90, 120), (600, 400)])
@pytest.mark.parametrize("nome", list(filtros.FILTROS))
def test_filtro_preserva_forma_e_tipo(nome, tamanho):
    img = np.random.default_rng(0).integers(0, 256, (*tamanho, 3), dtype=np.uint8)
    out = filtros.aplicar_filtro(nome, img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_original_e_intensidade_zero_nao_alteram():
    img = np.random.default_rng(1).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    assert np.array_equal(filtros.aplicar_filtro("Original", img), img)
    assert np.array_equal(filtros.aplicar_filtro("Noir", img, 0.0), img)
    assert not np.array_equal(filtros.aplicar_filtro("Noir", img, 1.0), img)


def test_filtros_pb_sao_cinza():
    img = np.random.default_rng(2).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    for nome in ("Moon", "Noir", "Lápis"):
        o = filtros.aplicar_filtro(nome, img)
        assert np.array_equal(o[:, :, 0], o[:, :, 1]) and np.array_equal(o[:, :, 1], o[:, :, 2])


def test_quente_mais_vermelho_que_frio():
    img = np.full((20, 20, 3), 128, np.uint8)
    q, f = filtros.aplicar_filtro("Quente", img), filtros.aplicar_filtro("Frio", img)
    assert q[0, 0, 2] > f[0, 0, 2] and q[0, 0, 0] < f[0, 0, 0]


def test_nao_altera_entrada():
    img = np.random.default_rng(3).integers(0, 256, (60, 60, 3), dtype=np.uint8)
    cp = img.copy()
    for nome in filtros.FILTROS:
        filtros.aplicar_filtro(nome, img, 0.5)
    assert np.array_equal(img, cp)
