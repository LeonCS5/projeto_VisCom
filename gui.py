"""Interface gráfica (tkinter) do projeto. Executar: python gui.py"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import processamento as p
import reconhecimento as rf

FILTROS_IMG = [("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")]


def ler_imagem(caminho):
    """imread que aceita caminhos com acentos no Windows."""
    try:
        dados = np.fromfile(caminho, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None


def gravar_imagem(caminho, img):
    ok, buf = cv2.imencode(os.path.splitext(caminho)[1] or ".png", img)
    if ok:
        buf.tofile(caminho)
    return ok


class Dialogo(tk.Toplevel):
    """Janela modal de parâmetros. `campos`: lista de (chave, rótulo, tipo, padrão[, opções]);
    tipo em {int, float, str, bool, 'opcao', 'arquivo', 'pasta', 'caminho'}."""

    def __init__(self, pai, titulo, campos):
        super().__init__(pai)
        self.title(titulo)
        self.transient(pai)
        self.resizable(False, False)
        self.resultado = None
        self.campos = campos
        self.vars = {}
        corpo = ttk.Frame(self, padding=12)
        corpo.pack()
        for i, c in enumerate(campos):
            chave, rotulo, tipo, padrao = c[:4]
            ttk.Label(corpo, text=rotulo).grid(row=i, column=0, sticky="w", pady=3, padx=(0, 10))
            if tipo is bool:
                v = tk.BooleanVar(value=padrao)
                ttk.Checkbutton(corpo, variable=v).grid(row=i, column=1, sticky="w")
            elif tipo == "opcao":
                v = tk.StringVar(value=padrao)
                ttk.Combobox(corpo, textvariable=v, values=c[4], state="readonly",
                             width=24).grid(row=i, column=1, sticky="w")
            elif tipo in ("arquivo", "pasta", "caminho"):
                v = tk.StringVar(value=padrao)
                linha = ttk.Frame(corpo)
                linha.grid(row=i, column=1, sticky="w")
                ttk.Entry(linha, textvariable=v, width=34).pack(side="left")
                if tipo in ("arquivo", "caminho"):
                    ttk.Button(linha, text="Arquivo…", command=lambda v=v: self._escolher(v, False)
                               ).pack(side="left", padx=2)
                if tipo in ("pasta", "caminho"):
                    ttk.Button(linha, text="Pasta…", command=lambda v=v: self._escolher(v, True)
                               ).pack(side="left")
            else:
                v = tk.StringVar(value=str(padrao))
                ttk.Entry(corpo, textvariable=v, width=14).grid(row=i, column=1, sticky="w")
            self.vars[chave] = v
        botoes = ttk.Frame(corpo)
        botoes.grid(row=len(campos), column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(botoes, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(botoes, text="Cancelar", command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        try:
            self.grab_set()
        except tk.TclError:  # janela ainda não mapeada (ex.: janela principal oculta)
            pass
        self.focus_set()

    def _escolher(self, var, pasta):
        r = filedialog.askdirectory(parent=self) if pasta else \
            filedialog.askopenfilename(parent=self, filetypes=FILTROS_IMG + [("Vídeo", "*.mp4 *.avi *.mov *.mkv")])
        if r:
            var.set(r)

    def _ok(self):
        saida = {}
        for c in self.campos:
            chave, rotulo, tipo = c[:3]
            bruto = self.vars[chave].get()
            try:
                saida[chave] = bruto if tipo in ("opcao", "arquivo", "pasta", "caminho", str) \
                    else (bool(bruto) if tipo is bool else tipo(bruto))
            except ValueError:
                messagebox.showerror("Valor inválido", f"Campo '{rotulo}' inválido.", parent=self)
                return
        self.resultado = saida
        self.destroy()


class Painel(ttk.LabelFrame):
    """Área que exibe uma imagem ajustada ao tamanho disponível."""

    def __init__(self, pai, titulo):
        super().__init__(pai, text=titulo)
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.img = None
        self._foto = None
        self.canvas.bind("<Configure>", lambda e: self.mostrar(self.img))

    def mostrar(self, img):
        self.img = img
        self.canvas.delete("all")
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cw, ch = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        esc = min(cw / rgb.shape[1], ch / rgb.shape[0])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * esc)), max(1, int(rgb.shape[0] * esc))),
                         interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(cw // 2, ch // 2, image=self._foto)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processamento de Imagens - Visão Computacional")
        self.geometry("1280x760")
        self.minsize(900, 560)
        self.orig = self.res = None
        self.caminho = None
        self.encadear = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Abra uma imagem para começar.")
        self._montar()

    # ---------- layout ----------
    def _montar(self):
        barra = ttk.Frame(self, padding=4)
        barra.pack(fill="x")
        for texto, cmd in (("Abrir…", self.abrir), ("Salvar resultado…", self.salvar),
                           ("Descartar processamento", self.resetar)):
            ttk.Button(barra, text=texto, command=cmd).pack(side="left", padx=2)
        ttk.Checkbutton(barra, text="Encadear (aplicar sobre o resultado anterior)",
                        variable=self.encadear).pack(side="left", padx=14)

        corpo = ttk.PanedWindow(self, orient="horizontal")
        corpo.pack(fill="both", expand=True, padx=4, pady=4)

        nb = ttk.Notebook(corpo, width=250)
        corpo.add(nb, weight=0)
        for secao, itens in self._menu():
            aba = ttk.Frame(nb, padding=6)
            nb.add(aba, text=secao)
            for rotulo, fn, livre in itens:
                ttk.Button(aba, text=rotulo, command=lambda f=fn, l=livre: self._executar(f, l)
                           ).pack(fill="x", pady=2)

        imagens = ttk.Frame(corpo)
        corpo.add(imagens, weight=1)
        imagens.columnconfigure((0, 1), weight=1, uniform="p")
        imagens.rowconfigure(0, weight=1)
        self.p_orig = Painel(imagens, "Original")
        self.p_res = Painel(imagens, "Processada")
        self.p_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        self.p_res.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=3
                  ).pack(fill="x", side="bottom")

    def _menu(self):
        return [
            ("Básico", [
                ("Inverter cores", self.a_inverter, False), ("Brilho e contraste", self.a_brilho, False),
                ("Somar imagens", self.a_somar, False), ("Subtrair imagens", self.a_subtrair, False),
                ("Níveis de cinza", self.a_cinza, False), ("RGB ⇄ HSV", self.a_hsv, False),
                ("Redimensionar", self.a_redimensionar, False), ("Recortar ROI", self.a_recortar, False)]),
            ("Filtros", [
                ("Filtro da média", self.a_media, False), ("Filtro da mediana", self.a_mediana, False),
                ("Filtro gaussiano", self.a_gauss, False), ("Bordas – Sobel", self.a_sobel, False),
                ("Bordas – Canny", self.a_canny, False)]),
            ("Histograma", [
                ("Histograma", self.a_histograma, False), ("Equalização", self.a_equalizar, False),
                ("Canais (split e análise)", self.a_canais, False), ("Binarização", self.a_binarizar, False),
                ("Quantização", self.a_quantizar, False)]),
            ("Segmentação", [
                ("Crescimento de regiões", self.a_regioes, False),
                ("Morfologia", self.a_morfologia, False)]),
            ("Faces", [
                ("Detectar faces", self.a_detectar, False),
                ("Cadastrar aluno (fotos)…", self.a_cadastrar, True),
                ("Treinar modelo", self.a_treinar, True),
                ("Reconhecer faces", self.a_reconhecer, False),
                ("Chamada automática…", self.a_chamada, True)]),
        ]

    # ---------- infraestrutura ----------
    def _executar(self, fn, livre):
        if not livre and self.orig is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        try:
            fn()
        except (cv2.error, ValueError, OSError, RuntimeError) as e:
            messagebox.showerror("Erro", str(e))

    def base(self):
        return self.res if self.encadear.get() and self.res is not None else self.orig

    def aplicar(self, img, nome):
        self.res = img
        self.p_res.mostrar(img)
        self.p_res.config(text=f"Processada – {nome}")
        h, w = img.shape[:2]
        self.status.set(f"{nome}  |  resultado {w}x{h}")

    def pedir(self, titulo, campos):
        d = Dialogo(self, titulo, campos)
        self.wait_window(d)
        return d.resultado

    def _abrir_imagem_extra(self):
        c = filedialog.askopenfilename(title="Segunda imagem", filetypes=FILTROS_IMG)
        if not c:
            return None
        img = ler_imagem(c)
        if img is None:
            raise ValueError("Não foi possível abrir a imagem.")
        return img

    # ---------- entrada / saída ----------
    def abrir(self):
        c = filedialog.askopenfilename(title="Abrir imagem", filetypes=FILTROS_IMG)
        if not c:
            return
        img = ler_imagem(c)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return
        self.orig, self.res, self.caminho = img, img.copy(), c
        self.p_orig.mostrar(self.orig)
        self.p_res.config(text="Processada")
        self.p_res.mostrar(self.res)
        self.status.set(f"{os.path.basename(c)}  |  {img.shape[1]}x{img.shape[0]}")

    def salvar(self):
        if self.res is None:
            messagebox.showinfo("Sem imagem", "Nada para salvar.")
            return
        c = filedialog.asksaveasfilename(defaultextension=".png", initialfile="saida.png",
                                         filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if c:
            self.status.set("Salvo em " + c if gravar_imagem(c, self.res) else "Falha ao salvar.")

    def resetar(self):
        if self.orig is not None:
            self.aplicar(self.orig.copy(), "Original restaurada")

    # ---------- operações básicas ----------
    def a_inverter(self):
        self.aplicar(p.inverter(self.base()), "Inversão")

    def a_brilho(self):
        r = self.pedir("Brilho e contraste", [("c", "Contraste (1.0 = igual)", float, 1.2),
                                              ("b", "Brilho (-255..255)", int, 30)])
        if r:
            self.aplicar(p.brilho_contraste(self.base(), r["c"], r["b"]), "Brilho/contraste")

    def a_somar(self):
        img, o = self.base(), self._abrir_imagem_extra()
        if o is None:
            return
        r = self.pedir("Soma", [("w", "Peso da segunda imagem (0..1)", float, 0.5)])
        if r:
            self.aplicar(p.somar(img, o, r["w"]), "Soma")

    def a_subtrair(self):
        img, o = self.base(), self._abrir_imagem_extra()
        if o is not None:
            self.aplicar(p.subtrair(img, o), "Subtração |A−B|")

    def a_cinza(self):
        self.aplicar(p.cinza(self.base()), "Níveis de cinza")

    def a_hsv(self):
        r = self.pedir("RGB ⇄ HSV", [("m", "Exibir", "opcao", "Imagem HSV",
                                      ["Imagem HSV", "Canal H", "Canal S", "Canal V", "HSV → BGR"])])
        if not r:
            return
        img, m = self.base(), r["m"]
        if m == "HSV → BGR":
            self.aplicar(p.hsv_para_bgr(img), m)
        elif m == "Imagem HSV":
            self.aplicar(p.para_hsv(img), "HSV (canais exibidos como cores)")
        else:
            self.aplicar(cv2.split(p.para_hsv(img))["HSV".index(m[-1])], m)

    def a_redimensionar(self):
        img = self.base()
        r = self.pedir("Redimensionar", [("e", f"Escala (atual {img.shape[1]}x{img.shape[0]})", float, 0.5),
                                         ("w", "…ou largura em px (0 = usar escala)", int, 0),
                                         ("h", "…e altura em px", int, 0)])
        if r:
            out = p.redimensionar(img, r["w"], r["h"]) if r["w"] and r["h"] else \
                p.redimensionar(img, escala=r["e"])
            self.aplicar(out, "Redimensionada")

    def a_recortar(self):
        img = self.base()
        messagebox.showinfo("ROI", "Na janela que abrir, arraste o retângulo e tecle ENTER (C cancela).")
        x, y, w, h = p.selecionar_roi(img)
        self.aplicar(p.recortar(img, x, y, w, h), "ROI")

    # ---------- filtros ----------
    def _kernel(self, titulo):
        r = self.pedir(titulo, [("k", "Tamanho do kernel (ímpar)", int, 5)])
        return r["k"] if r else None

    def a_media(self):
        k = self._kernel("Filtro da média")
        if k:
            self.aplicar(p.filtro_media(self.base(), k), f"Média {k}x{k}")

    def a_mediana(self):
        k = self._kernel("Filtro da mediana")
        if k:
            self.aplicar(p.filtro_mediana(self.base(), k), f"Mediana {k}x{k}")

    def a_gauss(self):
        r = self.pedir("Filtro gaussiano", [("k", "Tamanho do kernel (ímpar)", int, 5),
                                            ("s", "Sigma (0 = automático)", float, 0)])
        if r:
            self.aplicar(p.filtro_gaussiano(self.base(), r["k"], r["s"]), "Gaussiano")

    def a_sobel(self):
        r = self.pedir("Sobel", [("k", "Kernel (1, 3, 5 ou 7)", int, 3)])
        if r:
            self.aplicar(p.sobel(self.base(), r["k"]), "Sobel")

    def a_canny(self):
        r = self.pedir("Canny", [("a", "Limiar inferior", int, 100), ("b", "Limiar superior", int, 200)])
        if r:
            self.aplicar(p.canny(self.base(), r["a"], r["b"]), "Canny")

    # ---------- histograma e intensidade ----------
    def _janela_histogramas(self, titulo, imagens):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        cores = {"cinza": "k", "azul": "b", "verde": "g", "vermelho": "r"}
        j = tk.Toplevel(self)
        j.title(titulo)
        fig = Figure(figsize=(5 * len(imagens), 3.6), tight_layout=True)
        for i, (nome, im) in enumerate(imagens):
            ax = fig.add_subplot(1, len(imagens), i + 1)
            for canal, h in p.calcular_histogramas(im).items():
                ax.plot(h, color=cores[canal], label=canal)
            ax.set_xlim(0, 255)
            ax.set_title(nome)
            ax.set_xlabel("Intensidade")
            ax.set_ylabel("Pixels")
            ax.legend()
        FigureCanvasTkAgg(fig, master=j).get_tk_widget().pack(fill="both", expand=True)

    def a_histograma(self):
        self._janela_histogramas("Histograma", [("Histograma", self.base())])

    def a_equalizar(self):
        img = self.base()
        out = p.equalizar(img)
        self.aplicar(out, "Equalização")
        self._janela_histogramas("Histogramas antes e depois", [("Antes", img), ("Depois", out)])

    def a_canais(self):
        img = self.base()
        (_, _, _), montagem = p.separar_canais(img)
        self.aplicar(montagem, "Canais B | G | R")
        j = tk.Toplevel(self)
        j.title("Estatísticas por canal")
        tk.Label(j, text=p.estatisticas_canais(img), font=("Consolas", 10), justify="left",
                 padx=12, pady=8).pack()

    def a_binarizar(self):
        r = self.pedir("Binarização", [("m", "Método", "opcao", "otsu", ["otsu", "fixo", "adaptativo"]),
                                       ("l", "Limiar (método fixo)", int, 127)])
        if r:
            self.aplicar(p.binarizar(self.base(), r["m"], limiar=r["l"]), f"Binarização ({r['m']})")

    def a_quantizar(self):
        r = self.pedir("Quantização", [("n", "Níveis por canal (2..256)", int, 4)])
        if r:
            self.aplicar(p.quantizar(self.base(), r["n"]), f"Quantização ({r['n']} níveis)")

    # ---------- segmentação ----------
    def a_regioes(self):
        img = self.base()
        r = self.pedir("Crescimento de regiões", [("t", "Tolerância", int, 20)])
        if not r:
            return
        messagebox.showinfo("Semente", "Na janela que abrir, clique no ponto inicial (ESC cancela).")
        semente = p.selecionar_semente(img)
        if semente is None:
            return
        mascara = p.crescimento_regioes(img, semente, r["t"])
        self.aplicar(p.destacar_regiao(img, mascara), f"Região ({int((mascara > 0).sum())} px)")

    def a_morfologia(self):
        r = self.pedir("Morfologia", [
            ("op", "Operação", "opcao", "erosao", ["erosao", "dilatacao", "abertura", "fechamento"]),
            ("k", "Kernel", int, 3),
            ("f", "Formato", "opcao", "retangulo", ["retangulo", "elipse", "cruz"]),
            ("i", "Iterações", int, 1),
            ("bin", "Binarizar (Otsu) antes", bool, True)])
        if r:
            img = p.binarizar(self.base(), "otsu") if r["bin"] else self.base()
            self.aplicar(p.morfologia(img, r["op"], r["k"], r["f"], r["i"]), r["op"].capitalize())

    # ---------- reconhecimento facial ----------
    def a_detectar(self):
        out, n = rf.desenhar_deteccoes(self.base())
        self.aplicar(out, f"{n} rosto(s) detectado(s)")

    def a_cadastrar(self):
        r = self.pedir("Cadastrar aluno", [("nome", "Nome do aluno", str, ""),
                                           ("origem", "Foto ou pasta de fotos", "caminho", "")])
        if not r:
            return
        nome = r["nome"].strip().replace(" ", "_")
        if not nome or not r["origem"]:
            raise ValueError("Informe o nome e a origem das fotos.")
        ok, ruim = rf.cadastrar_de_fotos(nome, r["origem"])
        messagebox.showinfo("Cadastro", f"{ok} foto(s) aceita(s) para '{nome}'.\n"
                                        f"{ruim} rejeitada(s) (sem rosto, vários rostos ou ilegível).\n\n"
                                        "Treine o modelo para valer o cadastro.")

    def a_treinar(self):
        nomes, n = rf.treinar()
        messagebox.showinfo("Treinamento", f"Modelo treinado com {n} imagens de {len(nomes)} aluno(s):\n"
                                           + "\n".join(nomes))

    def a_reconhecer(self):
        out, achados = rf.reconhecer(self.base())
        self.aplicar(out, "Reconhecidos: " + (", ".join(achados) or "ninguém"))

    def a_chamada(self):
        r = self.pedir("Chamada automática", [
            ("fonte", "Foto, pasta de fotos ou vídeo", "caminho", ""),
            ("conf", "Reconhecimentos p/ marcar presença", int, 1)])
        if not r or not r["fonte"]:
            return
        pres, aus = rf.chamada(r["fonte"], confirmacoes=r["conf"], salvar_anotadas="chamada_anotada")
        j = tk.Toplevel(self)
        j.title("Resultado da chamada")
        tv = ttk.Treeview(j, columns=("aluno", "situacao"), show="headings", height=14)
        tv.heading("aluno", text="Aluno")
        tv.heading("situacao", text="Situação")
        tv.tag_configure("p", foreground="#1a7f37")
        tv.tag_configure("a", foreground="#b42318")
        for n in pres:
            tv.insert("", "end", values=(n, "Presente"), tags=("p",))
        for n in aus:
            tv.insert("", "end", values=(n, "Ausente"), tags=("a",))
        tv.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Label(j, text="Registro gravado em chamada.csv; imagens em chamada_anotada/").pack(pady=(0, 8))


if __name__ == "__main__":
    App().mainloop()
