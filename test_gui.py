"""Aciona todas as ações da GUI com diálogos simulados."""
import os
import tkinter as tk

import cv2
import numpy as np
import pytest

import gui
import processamento as p
import reconhecimento as rf


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        a = gui.App()
    except tk.TclError:
        pytest.skip("sem display")
    a.withdraw()
    rng = np.random.default_rng(0)
    cv2.imwrite("a.png", rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    cv2.imwrite("b.png", rng.integers(0, 256, (50, 50, 3), dtype=np.uint8))
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **k: "a.png")
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: "saida.png")
    for nome in ("showinfo", "showerror"):
        monkeypatch.setattr(gui.messagebox, nome, lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: 27)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (10, 10, 30, 30))
    monkeypatch.setattr(p, "selecionar_semente", lambda img: (20, 20))
    yield a
    a.destroy()


def com_respostas(monkeypatch, app, resp):
    """Faz app.pedir devolver `resp` (dict) em vez de abrir o diálogo."""
    monkeypatch.setattr(app, "pedir", lambda t, c: resp)


def test_abrir_e_salvar(app):
    app.abrir()
    assert app.orig.shape == (90, 120, 3)
    app.a_inverter()
    app.salvar()
    assert os.path.exists("saida.png")
    app.resetar()
    assert np.array_equal(app.res, app.orig)


def test_acoes_sem_imagem_nao_quebram(app):
    app._executar(app.a_inverter, False)  # mostra aviso, sem exceção
    assert app.res is None


@pytest.mark.parametrize("acao,resp", [
    ("a_brilho", {"c": 1.5, "b": 20}), ("a_somar", {"w": 0.3}),
    ("a_hsv", {"m": "Canal V"}), ("a_hsv", {"m": "Imagem HSV"}),
    ("a_redimensionar", {"e": 0.5, "w": 0, "h": 0}), ("a_redimensionar", {"e": 1, "w": 40, "h": 30}),
    ("a_media", {"k": 5}), ("a_mediana", {"k": 5}), ("a_gauss", {"k": 5, "s": 0.0}),
    ("a_sobel", {"k": 3}), ("a_canny", {"a": 100, "b": 200}),
    ("a_binarizar", {"m": "fixo", "l": 100}), ("a_quantizar", {"n": 4}),
    ("a_regioes", {"t": 30}),
    ("a_morfologia", {"op": "abertura", "k": 3, "f": "elipse", "i": 1, "bin": True}),
])
def test_acoes_com_parametros(app, monkeypatch, acao, resp):
    app.abrir()
    com_respostas(monkeypatch, app, resp)
    getattr(app, acao)()
    assert app.res is not None and app.p_res.img is app.res


def test_acoes_simples(app):
    app.abrir()
    for a in (app.a_cinza, app.a_recortar, app.a_histograma, app.a_equalizar,
              app.a_canais, app.a_detectar):
        a()
        app.update()


def test_encadear(app, monkeypatch):
    app.abrir()
    app.a_cinza()
    assert app.base() is app.orig
    app.encadear.set(True)
    assert app.base() is app.res


def test_dialogo_valida_entrada(app, monkeypatch):
    erros = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: erros.append(a))
    d = gui.Dialogo(app, "t", [("k", "Kernel", int, 5), ("m", "Modo", "opcao", "x", ["x", "y"])])
    d.vars["k"].set("abc")
    d._ok()
    assert erros and d.resultado is None
    d.vars["k"].set("7")
    d._ok()
    assert d.resultado == {"k": 7, "m": "x"}


def test_chamada_e_cadastro_pela_gui(app, monkeypatch):
    rng = np.random.default_rng(1)
    base = {nome: rng.integers(0, 256, (120, 120, 3), dtype=np.uint8) for nome in ("ana", "bia")}
    for nome, im in base.items():
        os.makedirs(f"fotos_{nome}")
        for i in range(3):
            cv2.imwrite(f"fotos_{nome}/{i}.png", np.clip(im.astype(int) + i, 0, 255).astype(np.uint8))
    os.makedirs("turma")
    cv2.imwrite("turma/0.png", base["ana"])
    monkeypatch.setattr(rf, "detectar_faces", lambda im, *a, **k: [(0, 0, im.shape[1], im.shape[0])])
    for nome in ("ana", "bia"):
        com_respostas(monkeypatch, app, {"nome": nome, "origem": f"fotos_{nome}"})
        app.a_cadastrar()
    app.a_treinar()
    com_respostas(monkeypatch, app, {"fonte": "turma", "conf": 1})
    app.a_chamada()
    with open("chamada.csv", encoding="utf-8") as f:
        assert "ana" in f.read()


def test_chamada_ao_vivo(app, monkeypatch):
    """Câmera simulada: um 'aluno' aparece nos quadros e deve ser marcado presente."""
    rng = np.random.default_rng(1)
    rosto = {n: rng.integers(0, 256, (120, 120, 3), dtype=np.uint8) for n in ("ana", "bia")}
    for nome, im in rosto.items():
        os.makedirs(f"faces/{nome}")
        for i in range(3):
            cv2.imwrite(f"faces/{nome}/{i}.png", np.clip(im.astype(int) + i, 0, 255).astype(np.uint8))
    monkeypatch.setattr(rf, "detectar_faces", lambda im, *a, **k: [(0, 0, im.shape[1], im.shape[0])])
    rf.treinar()

    class CamFalsa:
        def read(self):
            return True, rosto["bia"].copy()

        def release(self):
            pass

    monkeypatch.setattr(rf, "abrir_camera", lambda i: CamFalsa())
    com_respostas(monkeypatch, app, {"idx": 1, "conf": 3})
    app.a_chamada_camera()
    for _ in range(10):
        app.update()
        app.after(20)
    assert app._ao_vivo
    app._ao_vivo = None  # simula o encerramento
    with open("chamada.csv", encoding="utf-8") as f:
        linhas = f.read()
    assert "bia" in linhas and "ana" not in linhas.split("aluno")[1]
