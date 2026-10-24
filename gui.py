"""Interface gráfica (tkinter) do projeto. Executar: python gui.py"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import filtros
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


class Tooltip:
    """Balãozinho de ajuda que aparece ao parar o mouse sobre um widget."""

    def __init__(self, widget, texto, atraso=500):
        self.widget, self.texto, self.atraso = widget, texto, atraso
        self._id = self._janela = None
        widget.bind("<Enter>", self._agendar, add="+")
        widget.bind("<Leave>", self._fechar, add="+")
        widget.bind("<ButtonPress>", self._fechar, add="+")

    def _agendar(self, _e):
        self._fechar()
        self._id = self.widget.after(self.atraso, self._abrir)

    def _abrir(self):
        x, y = self.widget.winfo_rootx() + 12, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._janela = tk.Toplevel(self.widget)
        self._janela.wm_overrideredirect(True)
        self._janela.wm_geometry(f"+{x}+{y}")
        tk.Label(self._janela, text=self.texto, background="#fffbe6", relief="solid", borderwidth=1,
                 padx=6, pady=3, wraplength=260, justify="left").pack()

    def _fechar(self, _e=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self._janela:
            self._janela.destroy()
            self._janela = None


class Painel(ttk.LabelFrame):
    """Área que exibe uma imagem ajustada ao tamanho disponível. Sem imagem, mostra `vazio`."""

    def __init__(self, pai, titulo, vazio="", ao_clicar=None):
        super().__init__(pai, text=titulo)
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.img = None
        self.vazio = vazio
        self._foto = None
        self.canvas.bind("<Configure>", lambda e: self.mostrar(self.img))
        if ao_clicar:
            self.canvas.bind("<Button-1>", lambda e: ao_clicar())

    def mostrar(self, img):
        self.img = img
        self.canvas.delete("all")
        cw, ch = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        if img is None:
            self.canvas.create_text(cw // 2, ch // 2, text=self.vazio, fill="#9a9a9a",
                                    font=("Segoe UI", 12), justify="center", width=max(cw - 40, 40))
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        esc = min(cw / rgb.shape[1], ch / rgb.shape[0])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * esc)), max(1, int(rgb.shape[0] * esc))),
                         interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(cw // 2, ch // 2, image=self._foto)


AJUDA = """COMO USAR

1. Abra uma foto (Ctrl+O).
2. Na aba "✨ Estilos", clique em um filtro. A intensidade pode ser ajustada com o controle deslizante.
3. As outras abas têm as ferramentas de processamento: cada botão abre uma janelinha com os ajustes.
4. Compare o Original (esquerda) com o resultado (direita).
5. Errou? Ctrl+Z desfaz. "Original" volta ao começo.
6. Salve com Ctrl+S.

DICAS
• "Acumular efeitos": marcado, cada operação parte do resultado anterior (ex.: suavizar e depois aplicar um filtro).
  Desmarcado, cada operação parte sempre da imagem original.
• Passe o mouse sobre um botão para ver o que ele faz.
• A aba "Faces" faz cadastro, treino e chamada automática."""

TAM_MINIATURA = 92


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VisCom – Processamento de Imagens")
        self.geometry("1320x780")
        self.minsize(960, 600)
        self.orig = self.res = None
        self.caminho = None
        self.historico = []
        self.filtro_atual = None
        self._miniaturas = []
        self.encadear = tk.BooleanVar(value=False)
        self.intensidade = tk.DoubleVar(value=100)
        self.status = tk.StringVar(value="Bem-vindo! Abra uma imagem (Ctrl+O) para começar.")
        self._estilo()
        self._montar()
        self._atalhos()

    def _estilo(self):
        s = ttk.Style(self)
        s.configure(".", font=("Segoe UI", 10))
        s.configure("TButton", padding=(8, 5))
        s.configure("Acao.TButton", padding=(10, 8))
        s.configure("Filtro.TButton", padding=2, font=("Segoe UI", 9))
        s.configure("Titulo.TLabel", font=("Segoe UI", 11, "bold"))

    def _atalhos(self):
        self.bind("<Control-o>", lambda e: self.abrir())
        self.bind("<Control-s>", lambda e: self.salvar())
        self.bind("<Control-z>", lambda e: self.desfazer())

    # ---------- layout ----------
    def _montar(self):
        menu = tk.Menu(self)
        arquivo = tk.Menu(menu, tearoff=0)
        arquivo.add_command(label="Abrir…", accelerator="Ctrl+O", command=self.abrir)
        arquivo.add_command(label="Salvar resultado…", accelerator="Ctrl+S", command=self.salvar)
        arquivo.add_separator()
        arquivo.add_command(label="Sair", command=self.destroy)
        editar = tk.Menu(menu, tearoff=0)
        editar.add_command(label="Desfazer", accelerator="Ctrl+Z", command=self.desfazer)
        editar.add_command(label="Voltar à original", command=self.resetar)
        ajuda = tk.Menu(menu, tearoff=0)
        ajuda.add_command(label="Como usar", command=lambda: messagebox.showinfo("Como usar", AJUDA))
        menu.add_cascade(label="Arquivo", menu=arquivo)
        menu.add_cascade(label="Editar", menu=editar)
        menu.add_cascade(label="Ajuda", menu=ajuda)
        self.config(menu=menu)

        barra = ttk.Frame(self, padding=(6, 6))
        barra.pack(fill="x")
        for texto, cmd, dica in (
                ("📂  Abrir", self.abrir, "Abrir uma imagem (Ctrl+O)"),
                ("💾  Salvar", self.salvar, "Salvar o resultado em arquivo (Ctrl+S)"),
                ("↩  Desfazer", self.desfazer, "Desfaz a última alteração (Ctrl+Z)"),
                ("⟲  Original", self.resetar, "Descarta tudo e volta à imagem original")):
            b = ttk.Button(barra, text=texto, command=cmd, style="Acao.TButton")
            b.pack(side="left", padx=3)
            Tooltip(b, dica)
        c = ttk.Checkbutton(barra, text="Acumular efeitos", variable=self.encadear)
        c.pack(side="left", padx=18)
        Tooltip(c, "Marcado: cada efeito é aplicado sobre o resultado anterior.\n"
                   "Desmarcado: cada efeito parte da imagem original.")

        corpo = ttk.PanedWindow(self, orient="horizontal")
        corpo.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        nb = ttk.Notebook(corpo, width=375)
        corpo.add(nb, weight=0)
        nb.add(self._aba_estilos(nb), text="✨ Estilos")
        for secao, itens in self._menu():
            aba = ttk.Frame(nb, padding=8)
            nb.add(aba, text=secao)
            for rotulo, fn, livre, dica in itens:
                b = ttk.Button(aba, text=rotulo, command=lambda f=fn, l=livre: self._executar(f, l))
                b.pack(fill="x", pady=3)
                Tooltip(b, dica)

        imagens = ttk.Frame(corpo)
        corpo.add(imagens, weight=1)
        imagens.columnconfigure((0, 1), weight=1, uniform="p")
        imagens.rowconfigure(0, weight=1)
        self.p_orig = Painel(imagens, "Original", "Clique aqui ou tecle Ctrl+O\npara abrir uma imagem",
                             ao_clicar=lambda: self.orig is None and self.abrir())
        self.p_res = Painel(imagens, "Processada", "O resultado aparece aqui")
        self.p_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.p_res.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.p_orig.mostrar(None)
        self.p_res.mostrar(None)

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=4
                  ).pack(fill="x", side="bottom")

    def _aba_estilos(self, pai):
        aba = ttk.Frame(pai, padding=6)
        ttk.Label(aba, text="Escolha um estilo", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(aba, text="Clique em um filtro para aplicar.", foreground="#666").pack(anchor="w")

        inten = ttk.Frame(aba)
        inten.pack(fill="x", pady=(8, 6))
        ttk.Label(inten, text="Intensidade").pack(side="left")
        self.lbl_inten = ttk.Label(inten, text="100%", width=5, anchor="e")
        self.lbl_inten.pack(side="right")
        esc = ttk.Scale(inten, from_=0, to=100, variable=self.intensidade,
                        command=lambda v: self.lbl_inten.config(text=f"{float(v):.0f}%"))
        esc.pack(side="left", fill="x", expand=True, padx=8)
        esc.bind("<ButtonRelease-1>", lambda e: self._reaplicar_estilo())
        Tooltip(esc, "Quanto do filtro é aplicado: 0% = foto original, 100% = filtro completo.")

        area = ttk.Frame(aba)
        area.pack(fill="both", expand=True)
        self._canvas_est = tk.Canvas(area, highlightthickness=0)
        barra = ttk.Scrollbar(area, orient="vertical", command=self._canvas_est.yview)
        self._canvas_est.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self._canvas_est.pack(side="left", fill="both", expand=True)
        self.grade = ttk.Frame(self._canvas_est)
        self._canvas_est.create_window((0, 0), window=self.grade, anchor="nw")
        self.grade.bind("<Configure>",
                        lambda e: self._canvas_est.configure(scrollregion=self._canvas_est.bbox("all")))
        self._canvas_est.bind("<Enter>", lambda e: self._canvas_est.bind_all(
            "<MouseWheel>", lambda ev: self._canvas_est.yview_scroll(-1 * (ev.delta // 120), "units")))
        self._canvas_est.bind("<Leave>", lambda e: self._canvas_est.unbind_all("<MouseWheel>"))
        self._montar_grade()
        return aba

    def _montar_grade(self):
        """(Re)cria os botões de filtro; cada um mostra uma miniatura da imagem atual."""
        for w in self.grade.winfo_children():
            w.destroy()
        self._miniaturas = []
        if self.orig is not None:
            h, w = self.orig.shape[:2]
            lado = min(h, w)
            y0, x0 = (h - lado) // 2, (w - lado) // 2
            recorte = self.orig[y0:y0 + lado, x0:x0 + lado]
            base = cv2.resize(recorte, (TAM_MINIATURA, TAM_MINIATURA), interpolation=cv2.INTER_AREA)
        else:
            base = np.full((TAM_MINIATURA, TAM_MINIATURA, 3), 150, np.uint8)
            cv2.putText(base, "?", (34, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        for i, (nome, (_fn, dica)) in enumerate(filtros.FILTROS.items()):
            mini = filtros.aplicar_filtro(nome, base) if self.orig is not None else base
            foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(mini, cv2.COLOR_BGR2RGB)))
            self._miniaturas.append(foto)
            b = ttk.Button(self.grade, text=nome, image=foto, compound="top", style="Filtro.TButton",
                           command=lambda n=nome: self._aplicar_estilo(n))
            b.grid(row=i // 3, column=i % 3, padx=3, pady=3)
            Tooltip(b, dica)

    def _menu(self):
        return [
            ("Básico", [
                ("Inverter cores", self.a_inverter, False, "Gera o negativo da imagem."),
                ("Brilho e contraste", self.a_brilho, False, "Deixa a imagem mais clara/escura ou com mais/menos contraste."),
                ("Somar imagens", self.a_somar, False, "Mistura esta imagem com outra (escolha o quanto de cada)."),
                ("Subtrair imagens", self.a_subtrair, False, "Mostra a diferença entre esta imagem e outra."),
                ("Níveis de cinza", self.a_cinza, False, "Converte para preto e branco."),
                ("RGB ⇄ HSV", self.a_hsv, False, "Converte entre os modelos de cor e mostra os canais H, S e V."),
                ("Redimensionar", self.a_redimensionar, False, "Muda o tamanho da imagem, por escala ou em pixels."),
                ("Recortar região (ROI)", self.a_recortar, False, "Arraste o mouse sobre a área que deseja manter.")]),
            ("Realce", [
                ("Suavizar – média", self.a_media, False, "Desfoque simples; reduz ruído."),
                ("Suavizar – mediana", self.a_mediana, False, "Remove pontinhos de ruído preservando contornos."),
                ("Suavizar – gaussiano", self.a_gauss, False, "Desfoque suave e natural."),
                ("Bordas – Sobel", self.a_sobel, False, "Destaca contornos pela variação de brilho."),
                ("Bordas – Canny", self.a_canny, False, "Detector de bordas finas e precisas.")]),
            ("Análise", [
                ("Histograma", self.a_histograma, False, "Gráfico da distribuição de brilho/cores."),
                ("Equalizar histograma", self.a_equalizar, False, "Melhora o contraste distribuindo melhor os tons."),
                ("Canais de cor", self.a_canais, False, "Separa azul, verde e vermelho e mostra estatísticas."),
                ("Binarizar (preto/branco puro)", self.a_binarizar, False, "Limiar fixo, adaptativo ou Otsu."),
                ("Quantizar (menos cores)", self.a_quantizar, False, "Reduz a quantidade de níveis de cor.")]),
            ("Segmentar", [
                ("Crescimento de regiões", self.a_regioes, False, "Clique num ponto e a região de cor parecida é destacada."),
                ("Morfologia", self.a_morfologia, False, "Erosão, dilatação, abertura e fechamento.")]),
            ("Faces", [
                ("Detectar faces", self.a_detectar, False, "Marca os rostos encontrados na imagem."),
                ("Cadastrar aluno (fotos)…", self.a_cadastrar, True, "Guarda fotos de uma pessoa para o reconhecimento."),
                ("Treinar modelo", self.a_treinar, True, "Aprende os rostos cadastrados (refaça após cadastrar)."),
                ("Reconhecer faces", self.a_reconhecer, False, "Identifica quem aparece na imagem."),
                ("Chamada automática…", self.a_chamada, True, "Faz a chamada a partir de foto, pasta de fotos ou vídeo."),
                ("Chamada ao vivo (câmera)…", self.a_chamada_camera, True, "Marca presença em tempo real pela câmera.")]),
        ]

    # ---------- infraestrutura ----------
    def _executar(self, fn, livre):
        if not livre and self.orig is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro (Ctrl+O).")
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            fn()
        except (cv2.error, ValueError, OSError, RuntimeError) as e:
            messagebox.showerror("Erro", str(e))
        finally:
            self.config(cursor="")

    def base(self):
        return self.res if self.encadear.get() and self.res is not None else self.orig

    def aplicar(self, img, nome):
        if self.res is not None:
            self.historico.append(self.res)
            del self.historico[:-30]
        self._mostrar_resultado(img, nome)

    def _mostrar_resultado(self, img, nome):
        self.res = img
        self.p_res.mostrar(img)
        self.p_res.config(text=f"Processada – {nome}")
        h, w = img.shape[:2]
        self.status.set(f"{nome}  |  resultado {w}x{h}")

    def desfazer(self):
        if not self.historico:
            self.status.set("Nada para desfazer.")
            return
        self._mostrar_resultado(self.historico.pop(), "desfeito")

    def _aplicar_estilo(self, nome):
        if self.orig is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro (Ctrl+O).")
            return
        self.filtro_atual = nome
        self._executar(lambda: self._estilo_agora(nome, push=True), True)

    def _reaplicar_estilo(self):
        if self.filtro_atual and self.orig is not None:
            self._executar(lambda: self._estilo_agora(self.filtro_atual, push=False), True)

    def _estilo_agora(self, nome, push):
        out = filtros.aplicar_filtro(nome, self.base(), self.intensidade.get() / 100)
        (self.aplicar if push else self._mostrar_resultado)(out, f"estilo {nome}")

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
        self.historico.clear()
        self.filtro_atual = None
        self.p_orig.mostrar(self.orig)
        self.p_res.config(text="Processada")
        self.p_res.mostrar(self.res)
        self._montar_grade()
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

    # ---------- chamada ao vivo ----------
    def a_chamada_camera(self):
        if getattr(self, "_ao_vivo", None):
            messagebox.showinfo("Chamada", "Já existe uma chamada em andamento.")
            return
        r = self.pedir("Chamada ao vivo", [
            ("idx", "Câmera: índice (0, 1…) ou IP do celular", str, "192.168.18.4"),
            ("conf", "Quadros de reconhecimento p/ marcar presença", int, 5)])
        if not r:
            return
        sessao = rf.SessaoChamada(confirmacoes=r["conf"])
        cap = rf.abrir_camera(r["idx"])

        painel = tk.Toplevel(self)
        painel.title("Chamada ao vivo")
        lista = tk.Listbox(painel, width=34, height=14)
        lista.pack(fill="both", expand=True, padx=8, pady=8)
        info = tk.StringVar()
        ttk.Label(painel, textvariable=info).pack()
        self._ao_vivo = True

        def encerrar():
            self._ao_vivo = None
            cap.release()
            painel.destroy()
            self.status.set(f"Chamada encerrada. Presentes: {', '.join(sorted(sessao.presentes)) or '-'}  |  "
                            f"Ausentes: {', '.join(sessao.ausentes) or '-'}. Registro em chamada.csv")

        ttk.Button(painel, text="Encerrar chamada", command=encerrar).pack(pady=8)
        painel.protocol("WM_DELETE_WINDOW", encerrar)

        def passo():
            if not self._ao_vivo:
                return
            ok, quadro = cap.read()
            if ok:
                vis, novos = sessao.processar(quadro)
                self.p_orig.mostrar(quadro)
                self.p_res.mostrar(vis)
                for n in novos:
                    lista.insert("end", f"✔ {n}")
                info.set(f"Presentes: {len(sessao.presentes)}/{len(sessao.nomes)}")
                self.status.set("Chamada ao vivo em andamento…")
            self.after(15, passo)

        passo()


if __name__ == "__main__":
    App().mainloop()
