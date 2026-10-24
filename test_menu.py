"""Percorre o menu inteiro com entradas simuladas e janelas do OpenCV neutralizadas."""
import builtins
import os

import cv2
import numpy as np
import pytest

import main
import reconhecimento as rf


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 27)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (10, 10, 30, 30))
    monkeypatch.setattr(main.p, "selecionar_semente", lambda img: (20, 20))
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    cv2.imwrite("a.png", rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    cv2.imwrite("b.png", rng.integers(0, 256, (50, 50, 3), dtype=np.uint8))
    main.estado.update(orig=None, res=None, caminho=None)
    return tmp_path


def roda(monkeypatch, entradas):
    it = iter(entradas)
    monkeypatch.setattr(builtins, "input", lambda *_: next(it))
    main.main()


def n(nome):
    """Número da opção do menu cujo rótulo começa com `nome`."""
    i = 1
    for _, itens in main.MENU:
        for rotulo, _fn in itens:
            if rotulo.startswith(nome):
                return str(i)
            i += 1
    raise KeyError(nome)


def test_menu_completo(ambiente, monkeypatch, capsys):
    ent = [n("Abrir"), "a.png"]
    ent += [n("Salvar"), "saida.png", n("Ver"), n("Descartar")]
    ent += [n("Inverter"), "n", n("Brilho"), "n", "1.5", "20"]
    ent += [n("Somar"), "n", "b.png", "0.3", n("Subtrair"), "n", "b.png"]
    ent += [n("Níveis"), "n", n("RGB"), "n", "s", "s"]
    ent += [n("Redimensionar"), "n", "e", "0.5", n("Recortar"), "n", "s"]
    ent += [n("Filtro da média"), "n", "5", n("Filtro da mediana"), "n", "5",
            n("Filtro gaussiano"), "n", "5", "0"]
    ent += [n("Bordas - Sobel"), "n", "3", n("Bordas - Canny"), "n", "100", "200"]
    ent += [n("Histograma"), "n", n("Equalização"), "n", "n", n("Split"), "n",
            n("Binarização"), "n", "fixo", "100", n("Quantização"), "n", "4"]
    ent += [n("Crescimento"), "n", "30", n("Morfologia"), "n", "s", "abertura", "3", "elipse", "1"]
    ent += [n("Detectar"), "n", "0"]
    roda(monkeypatch, ent)
    out = capsys.readouterr().out
    assert os.path.exists("saida.png")
    assert "inválid" not in out.lower() and "erro" not in out.lower()


def test_menu_sem_imagem_e_opcao_invalida(ambiente, monkeypatch, capsys):
    roda(monkeypatch, ["5", "99", "0"])
    out = capsys.readouterr().out
    assert "Nenhuma imagem carregada" in out and "Opção inválida" in out


def test_treino_e_chamada_por_fotos(ambiente, monkeypatch, capsys):
    rng = np.random.default_rng(1)
    base = {nome: rng.integers(0, 256, (120, 120, 3), dtype=np.uint8) for nome in ("ana", "bia")}
    for nome, im in base.items():
        os.makedirs(f"fotos_{nome}")
        for i in range(3):
            cv2.imwrite(f"fotos_{nome}/{i}.png", np.clip(im.astype(int) + i, 0, 255).astype(np.uint8))
    os.makedirs("turma")
    cv2.imwrite("turma/0.png", base["ana"])
    monkeypatch.setattr(rf, "detectar_faces", lambda im, *a, **k: [(0, 0, im.shape[1], im.shape[0])])
    roda(monkeypatch, [n("Cadastrar"), "ana", "fotos_ana", n("Cadastrar"), "bia", "fotos_bia", n("Treinar"),
                       n("Chamada"), "turma", "1", "0"])
    out = capsys.readouterr().out
    assert "Presentes: ana" in out and "Ausentes:  bia" in out
    assert os.path.exists("chamada.csv") and os.path.exists("chamada_anotada/0000.jpg")
