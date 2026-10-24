"""Menu principal do projeto de Visão Computacional."""
import os

import cv2
import numpy as np

import processamento as p
import reconhecimento as rf

estado = {"orig": None, "res": None, "caminho": None}


# ---------- utilidades de interface ----------

def perguntar(msg, tipo=str, padrao=None):
    sufixo = f" [{padrao}]" if padrao is not None else ""
    while True:
        txt = input(f"{msg}{sufixo}: ").strip()
        if not txt and padrao is not None:
            return padrao
        try:
            return tipo(txt)
        except ValueError:
            print("Valor inválido.")


def mostrar(*imagens, titulo="Original | Processada", altura_max=600):
    """Exibe as imagens lado a lado numa única janela."""
    itens = []
    for im in imagens:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
        esc = min(1.0, altura_max / im.shape[0])
        itens.append(cv2.resize(im, None, fx=esc, fy=esc))
    h = max(i.shape[0] for i in itens)
    itens = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in itens]
    cv2.imshow(titulo, np.hstack(itens))
    print("(tecle qualquer tecla na janela para continuar)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exige_imagem():
    if estado["orig"] is None:
        print("Nenhuma imagem carregada. Use a opção 1.")
        return False
    return True


def base():
    """Imagem de partida: o último resultado, se o usuário quiser encadear."""
    if estado["res"] is not None and perguntar("Aplicar sobre o resultado anterior? (s/n)", str, "n") == "s":
        return estado["res"]
    return estado["orig"]


def aplicar(resultado, titulo="Resultado"):
    estado["res"] = resultado
    mostrar(estado["orig"], resultado, titulo=titulo)


def ler_outra(msg="Caminho da segunda imagem"):
    img = cv2.imread(perguntar(msg))
    if img is None:
        print("Não foi possível abrir a imagem.")
    return img


# ---------- ações do menu ----------

def abrir():
    caminho = perguntar("Caminho da imagem")
    img = cv2.imread(caminho)
    if img is None:
        print("Não foi possível abrir a imagem.")
        return
    estado.update(orig=img, res=img.copy(), caminho=caminho)  # cópia independente
    print(f"Carregada: {img.shape[1]}x{img.shape[0]}, {img.shape[2]} canais.")
    mostrar(img, titulo="Imagem carregada")


def salvar():
    if not exige_imagem():
        return
    destino = perguntar("Arquivo de saída", str, "saida.png")
    print("Salvo." if cv2.imwrite(destino, estado["res"]) else "Falha ao salvar.")


def ver():
    if exige_imagem():
        mostrar(estado["orig"], estado["res"])


def resetar():
    if exige_imagem():
        estado["res"] = estado["orig"].copy()
        print("Resultado restaurado para a original.")


def a_inverter():
    aplicar(p.inverter(base()), "Inversão")


def a_brilho():
    img = base()
    c = perguntar("Contraste (1.0 = igual)", float, 1.2)
    b = perguntar("Brilho (-255..255)", int, 30)
    aplicar(p.brilho_contraste(img, c, b), "Brilho/contraste")


def a_somar():
    img = base()
    o = ler_outra()
    if o is not None:
        aplicar(p.somar(img, o, perguntar("Peso da segunda (0..1)", float, 0.5)), "Soma")


def a_subtrair():
    img = base()
    o = ler_outra()
    if o is not None:
        aplicar(p.subtrair(img, o), "Subtração (|A-B|)")


def a_cinza():
    aplicar(p.cinza(base()), "Cinza")


def a_hsv():
    aplicar(p.para_hsv(base()), "HSV (exibido como BGR)")
    if perguntar("Mostrar canais H, S, V separados? (s/n)", str, "n") == "s":
        h, s, v = cv2.split(estado["res"])
        mostrar(h, s, v, titulo="H | S | V")
    if perguntar("Converter de volta para BGR? (s/n)", str, "n") == "s":
        aplicar(p.hsv_para_bgr(estado["res"]), "HSV -> BGR")


def a_redimensionar():
    img = base()
    print(f"Tamanho atual: {img.shape[1]}x{img.shape[0]}")
    if perguntar("Por escala (e) ou largura/altura (t)?", str, "e") == "e":
        out = p.redimensionar(img, escala=perguntar("Escala", float, 0.5))
    else:
        out = p.redimensionar(img, perguntar("Largura", int), perguntar("Altura", int))
    aplicar(out, "Redimensionada")
    print(f"Novo tamanho: {out.shape[1]}x{out.shape[0]}")


def a_recortar():
    img = base()
    if perguntar("Selecionar com o mouse? (s/n)", str, "s") == "s":
        x, y, w, h = p.selecionar_roi(img)
    else:
        x, y = perguntar("x", int), perguntar("y", int)
        w, h = perguntar("largura", int), perguntar("altura", int)
    try:
        aplicar(p.recortar(img, x, y, w, h), "ROI")
    except ValueError as e:
        print(e)


def a_media():
    img = base()
    aplicar(p.filtro_media(img, perguntar("Tamanho do kernel", int, 5)), "Média")


def a_mediana():
    img = base()
    aplicar(p.filtro_mediana(img, perguntar("Tamanho do kernel", int, 5)), "Mediana")


def a_gauss():
    img = base()
    aplicar(p.filtro_gaussiano(img, perguntar("Tamanho do kernel", int, 5),
                               perguntar("Sigma (0 = automático)", float, 0)), "Gaussiano")


def a_sobel():
    img = base()
    aplicar(p.sobel(img, perguntar("Kernel (1,3,5,7)", int, 3)), "Sobel")


def a_canny():
    img = base()
    aplicar(p.canny(img, perguntar("Limiar inferior", int, 100),
                    perguntar("Limiar superior", int, 200)), "Canny")


def a_histograma():
    p.mostrar_histograma(base(), "Histograma")


def a_equalizar():
    img = base()
    out = p.equalizar(img)
    aplicar(out, "Equalização")
    if perguntar("Comparar histogramas? (s/n)", str, "n") == "s":
        p.mostrar_histograma(img, "Antes")
        p.mostrar_histograma(out, "Depois")


def a_canais():
    img = base()
    print("Estatísticas por canal:\n" + p.estatisticas_canais(img))
    _, montagem = p.separar_canais(img)
    aplicar(montagem, "Canais B | G | R")


def a_binarizar():
    img = base()
    m = perguntar("Método (fixo/adaptativo/otsu)", str, "otsu")
    if m == "fixo":
        out = p.binarizar(img, "fixo", limiar=perguntar("Limiar", int, 127))
    else:
        out = p.binarizar(img, m)
    aplicar(out, f"Binarização ({m})")


def a_quantizar():
    img = base()
    aplicar(p.quantizar(img, perguntar("Número de níveis (2..256)", int, 4)), "Quantização")


def a_regioes():
    img = base()
    print("Clique na semente na janela que abrirá.")
    semente = p.selecionar_semente(img)
    if semente is None:
        print("Cancelado.")
        return
    try:
        mascara = p.crescimento_regioes(img, semente, perguntar("Tolerância", int, 20))
    except ValueError as e:
        print(e)
        return
    aplicar(p.destacar_regiao(img, mascara), "Crescimento de regiões")


def a_morfologia():
    img = base()
    if perguntar("Binarizar (Otsu) antes? (s/n)", str, "s") == "s":
        img = p.binarizar(img, "otsu")
    op = perguntar("Operação (erosao/dilatacao/abertura/fechamento)", str, "erosao")
    if op not in ("erosao", "dilatacao", "abertura", "fechamento"):
        print("Operação inválida.")
        return
    aplicar(p.morfologia(img, op, perguntar("Kernel", int, 3),
                         perguntar("Formato (retangulo/elipse/cruz)", str, "retangulo"),
                         perguntar("Iterações", int, 1)), op.capitalize())


def a_detectar_faces():
    out, n = rf.desenhar_deteccoes(base())
    print(f"{n} rosto(s) detectado(s).")
    aplicar(out, "Detecção de faces")


def a_treinar():
    try:
        nomes, n = rf.treinar()
        print(f"Treinado com {n} imagens de {len(nomes)} aluno(s): {', '.join(nomes)}")
    except (OSError, RuntimeError) as e:
        print(e)


def a_cadastrar():
    nome = perguntar("Nome do aluno").strip().replace(" ", "_")
    origem = perguntar("Foto ou pasta com fotos do aluno (1 rosto por foto)")
    ok, ruim = rf.cadastrar_de_fotos(nome, origem)
    print(f"{ok} foto(s) aceita(s) em faces/{nome}; {ruim} rejeitada(s) (sem rosto, vários rostos ou ilegível).")
    if ok:
        print("Lembre-se de treinar o modelo.")


def a_reconhecer_imagem():
    try:
        out, achados = rf.reconhecer(base())
    except (OSError, RuntimeError) as e:
        print(e)
        return
    print("Reconhecidos:", ", ".join(achados) or "ninguém")
    aplicar(out, "Reconhecimento facial")


def a_chamada():
    fonte = perguntar("Foto, pasta de fotos ou vídeo da turma")
    conf = perguntar("Reconhecimentos necessários para marcar presença", int, 1)
    try:
        pres, aus = rf.chamada(fonte, confirmacoes=conf, salvar_anotadas="chamada_anotada")
    except (OSError, RuntimeError) as e:
        print(e)
        return
    print("Presentes:", ", ".join(pres) or "-")
    print("Ausentes: ", ", ".join(aus) or "-")
    print("Registro em chamada.csv; imagens anotadas em chamada_anotada/")


MENU = [
    ("ENTRADA / SAÍDA", [
        ("Abrir imagem", abrir), ("Salvar resultado", salvar),
        ("Ver original x processada", ver), ("Descartar processamento (voltar à original)", resetar)]),
    ("OPERAÇÕES BÁSICAS", [
        ("Inverter cores", a_inverter), ("Brilho e contraste", a_brilho),
        ("Somar imagens", a_somar), ("Subtrair imagens", a_subtrair),
        ("Níveis de cinza", a_cinza), ("RGB <-> HSV", a_hsv),
        ("Redimensionar", a_redimensionar), ("Recortar ROI", a_recortar)]),
    ("FILTRAGEM E REALCE", [
        ("Filtro da média", a_media), ("Filtro da mediana", a_mediana),
        ("Filtro gaussiano", a_gauss), ("Bordas - Sobel", a_sobel), ("Bordas - Canny", a_canny)]),
    ("HISTOGRAMA E INTENSIDADE", [
        ("Histograma", a_histograma), ("Equalização", a_equalizar),
        ("Split/análise de canais", a_canais), ("Binarização", a_binarizar),
        ("Quantização", a_quantizar)]),
    ("SEGMENTAÇÃO E MORFOLOGIA", [
        ("Crescimento de regiões", a_regioes),
        ("Morfologia (erosão/dilatação/abertura/fechamento)", a_morfologia)]),
    ("RECONHECIMENTO FACIAL", [
        ("Detectar faces", a_detectar_faces), ("Cadastrar aluno (fotos)", a_cadastrar),
        ("Treinar modelo (pasta faces/)", a_treinar),
        ("Reconhecer faces na imagem", a_reconhecer_imagem),
        ("Chamada automática (fotos/vídeo)", a_chamada)]),
]


def main():
    acoes = {}
    while True:
        print("\n===== PROCESSAMENTO DE IMAGENS =====")
        if estado["caminho"]:
            print(f"Imagem: {estado['caminho']}")
        n = 1
        for secao, itens in MENU:
            print(f"-- {secao}")
            for nome, fn in itens:
                acoes[n] = fn
                print(f"  {n:2d}) {nome}")
                n += 1
        print("   0) Sair")
        escolha = perguntar("Opção", int, -1)
        if escolha == 0:
            break
        fn = acoes.get(escolha)
        if fn is None:
            print("Opção inválida.")
            continue
        # tudo exceto abrir (1) e cadastro/treino/chamada exige imagem carregada
        sem_imagem = {abrir, a_treinar, a_cadastrar, a_chamada}
        if fn not in sem_imagem and not exige_imagem():
            continue
        try:
            fn()
        except cv2.error as e:
            print(f"Erro do OpenCV: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
