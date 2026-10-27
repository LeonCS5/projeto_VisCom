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


class Formulario(ttk.Frame):
    """Formulário de parâmetros que fica dentro do painel lateral (sem janela nova).

    `campos`: lista de (chave, rótulo, tipo, padrão[, extra]).
      - int/float com extra=(mín, máx[, passo])  -> barra deslizante + caixa numérica
      - int/float sem extra / str                 -> caixa de texto
      - bool                                      -> caixa de seleção
      - 'opcao' com extra=[opções]                -> lista de escolha
      - 'arquivo' | 'pasta' | 'caminho'           -> caixa + botões de procurar
    `ao_mudar` é chamado a cada alteração (para a prévia ao vivo)."""

    def __init__(self, pai, titulo, campos, ao_mudar=None):
        super().__init__(pai)
        self.campos = campos
        self.ao_mudar = ao_mudar or (lambda: None)
        self.resultado = None
        self.concluido = tk.IntVar(value=0)
        self._leitores = {}
        self._setters = {}
        self._repor = []
        ttk.Label(self, text=titulo, style="Titulo.TLabel").pack(anchor="w", pady=(0, 4))
        for c in campos:
            linha = ttk.Frame(self)
            linha.pack(fill="x", pady=2)
            self._campo(linha, *c[:4], c[4] if len(c) > 4 else None)
        botoes = ttk.Frame(self)
        botoes.pack(fill="x", pady=(8, 0))
        ttk.Button(botoes, text="✔ Aplicar", style="Destaque.TButton", command=self._ok
                   ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(botoes, text="↺", width=3, command=self._padrao).pack(side="left", padx=3)
        ttk.Button(botoes, text="✖ Cancelar", command=self._cancelar
                   ).pack(side="left", expand=True, fill="x", padx=(3, 0))

    # ----- construção dos campos -----
    def _campo(self, linha, chave, rotulo, tipo, padrao, extra):
        if tipo in (int, float) and extra:
            self._slider(linha, chave, rotulo, tipo, padrao, extra)
        elif tipo is bool:
            v = tk.BooleanVar(value=padrao)
            ttk.Checkbutton(linha, text=rotulo, variable=v, command=self.ao_mudar).pack(anchor="w")
            self._leitores[chave] = v.get
            self._setters[chave] = v.set
            self._repor.append(lambda v=v, p=padrao: v.set(p))
        elif tipo == "opcao":
            ttk.Label(linha, text=rotulo).pack(side="left")
            v = tk.StringVar(value=padrao)
            cb = ttk.Combobox(linha, textvariable=v, values=extra, state="readonly", width=16)
            cb.pack(side="right")
            cb.bind("<<ComboboxSelected>>", lambda e: self.ao_mudar())
            self._leitores[chave] = v.get
            self._setters[chave] = v.set
            self._repor.append(lambda v=v, p=padrao: v.set(p))
        elif tipo in ("arquivo", "pasta", "caminho"):
            ttk.Label(linha, text=rotulo).pack(anchor="w")
            v = tk.StringVar(value=padrao)
            ttk.Entry(linha, textvariable=v).pack(fill="x")
            bt = ttk.Frame(linha)
            bt.pack(anchor="w", pady=2)
            if tipo in ("arquivo", "caminho"):
                ttk.Button(bt, text="Arquivo…", command=lambda v=v: self._procurar(v, False)).pack(side="left")
            if tipo in ("pasta", "caminho"):
                ttk.Button(bt, text="Pasta…", command=lambda v=v: self._procurar(v, True)).pack(side="left", padx=4)
            self._leitores[chave] = v.get
            self._setters[chave] = v.set
            self._repor.append(lambda v=v, p=padrao: v.set(p))
        else:
            ttk.Label(linha, text=rotulo).pack(side="left")
            v = tk.StringVar(value=str(padrao))
            ttk.Entry(linha, textvariable=v, width=14).pack(side="right")
            self._leitores[chave] = lambda v=v, t=tipo, r=rotulo: self._converter(v.get(), t, r)
            self._setters[chave] = v.set
            self._repor.append(lambda v=v, p=padrao: v.set(str(p)))

    def _slider(self, linha, chave, rotulo, tipo, padrao, extra):
        minimo, maximo = extra[0], extra[1]
        passo = extra[2] if len(extra) > 2 else (1 if tipo is int else 0.1)
        casas = 0 if tipo is int else max(0, -int(np.floor(np.log10(passo))))
        estado = {"x": float(padrao), "trava": False}

        def ajustar(x):
            x = minimo + round((x - minimo) / passo) * passo
            return round(min(maximo, max(minimo, x)), 6)

        def texto(x):
            return f"{x:.{casas}f}"

        topo = ttk.Frame(linha)
        topo.pack(fill="x")
        ttk.Label(topo, text=rotulo).pack(side="left")
        sv = tk.StringVar(value=texto(padrao))
        caixa = ttk.Spinbox(topo, from_=minimo, to=maximo, increment=passo, textvariable=sv, width=7)
        caixa.pack(side="right")
        escala = ttk.Scale(linha, from_=minimo, to=maximo)
        escala.set(padrao)
        escala.pack(fill="x")

        def definir(x, origem_escala):
            x = ajustar(x)
            if x == estado["x"]:
                sv.set(texto(x))
                return
            estado["x"] = x
            sv.set(texto(x))
            estado["trava"] = True
            escala.set(x)
            estado["trava"] = False
            self.ao_mudar()

        escala.configure(command=lambda v: None if estado["trava"] else definir(float(v), True))

        def da_caixa(_e=None):
            try:
                definir(float(sv.get().replace(",", ".")), False)
            except ValueError:
                sv.set(texto(estado["x"]))

        for ev in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            caixa.bind(ev, lambda e: self.after_idle(da_caixa))
        self._leitores[chave] = lambda: int(round(estado["x"])) if tipo is int else estado["x"]
        self._setters[chave] = lambda x: definir(x, False)
        self._repor.append(lambda p=padrao: definir(p, False))

    # ----- ações -----
    @staticmethod
    def _converter(bruto, tipo, rotulo):
        try:
            return bruto if tipo is str else tipo(bruto.replace(",", "."))
        except ValueError:
            raise ValueError(f"Valor inválido em '{rotulo}'.")

    def _procurar(self, var, pasta):
        r = filedialog.askdirectory() if pasta else filedialog.askopenfilename(
            filetypes=FILTROS_IMG + [("Vídeo", "*.mp4 *.avi *.mov *.mkv")])
        if r:
            var.set(r)

    def definir(self, chave, valor):
        """Altera um campo por código (equivale ao usuário mexer no controle)."""
        self._setters[chave](valor)
        self.ao_mudar()

    def valores(self):
        return {chave: leitor() for chave, leitor in self._leitores.items()}

    def _ok(self):
        try:
            self.resultado = self.valores()
        except ValueError as e:
            messagebox.showerror("Valor inválido", str(e))
            return
        self.concluido.set(1)

    def _cancelar(self):
        self.resultado = None
        self.concluido.set(1)

    def _padrao(self):
        for f in self._repor:
            f()
        self.ao_mudar()


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
        self.geometry("1320x840")
        self.minsize(960, 600)
        self.orig = self.res = None
        self.caminho = None
        self.historico = []
        self.filtro_atual = None
        self._miniaturas = []
        self._ao_vivo = None
        self._encerrar_chamada = lambda: None
        self._host = None  # painel de ajustes onde os formulários aparecem
        self._pedindo = False
        self._titulo_res = "Processada"
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
        s.configure("Ferramenta.TButton", padding=(4, 5))
        s.configure("Destaque.TButton", padding=(6, 5), font=("Segoe UI", 10, "bold"))
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
        for secao, itens in self._menu():
            nb.add(self._aba_ferramentas(nb, itens), text=secao)
        nb.add(self._aba_estilos(nb), text="✨ Estilos")

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

    def _aba_ferramentas(self, pai, itens):
        """Aba com os botões das ferramentas (2 colunas) e, embaixo, a área de ajustes."""
        aba = ttk.Frame(pai, padding=8)
        botoes = ttk.Frame(aba)
        botoes.pack(fill="x")
        botoes.columnconfigure((0, 1), weight=1, uniform="b")
        ajustes = ttk.LabelFrame(aba, text="Ajustes", padding=8)
        ajustes.pack(fill="both", expand=True, pady=(10, 0))
        self._placeholder(ajustes)
        for i, (rotulo, fn, livre, dica) in enumerate(itens):
            b = ttk.Button(botoes, text=rotulo, style="Ferramenta.TButton",
                           command=lambda f=fn, l=livre, h=ajustes: self._executar(f, l, h))
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            Tooltip(b, dica)
        if self._host is None:
            self._host = ajustes
        return aba

    @staticmethod
    def _placeholder(host):
        ttk.Label(host, text="Escolha uma ferramenta acima.\nOs ajustes aparecem aqui.",
                  foreground="#888", justify="center").pack(expand=True, pady=20)

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
                ("Brilho/contraste", self.a_brilho, False, "Deixa a imagem mais clara/escura ou com mais/menos contraste."),
                ("Saturação", self.a_saturacao, False, "Intensifica ou apaga as cores."),
                ("Níveis de cinza", self.a_cinza, False, "Converte para preto e branco."),
                ("RGB ⇄ HSV", self.a_hsv, False, "Converte entre os modelos de cor e mostra os canais H, S e V."),
                ("Somar imagens", self.a_somar, False, "Mistura esta imagem com outra (escolha o quanto de cada)."),
                ("Subtrair imagens", self.a_subtrair, False, "Mostra a diferença entre esta imagem e outra."),
                ("Redimensionar", self.a_redimensionar, False, "Muda o tamanho da imagem, por escala ou em pixels."),
                ("Recortar (ROI)", self.a_recortar, False, "Escolhe a região a manter, por barras ou arrastando o mouse."),
                ("Rotacionar", self.a_rotacionar, False, "Gira a imagem em qualquer ângulo."),
                ("Espelhar", self.a_espelhar, False, "Inverte a imagem na horizontal, vertical ou ambos.")]),
            ("Realce", [
                ("Média", self.a_media, False, "Desfoque simples; reduz ruído."),
                ("Mediana", self.a_mediana, False, "Remove pontinhos de ruído preservando contornos."),
                ("Gaussiano", self.a_gauss, False, "Desfoque suave e natural."),
                ("Nitidez", self.a_nitidez, False, "Aumenta a definição dos detalhes."),
                ("Bordas – Sobel", self.a_sobel, False, "Destaca contornos pela variação de brilho."),
                ("Bordas – Canny", self.a_canny, False, "Detector de bordas finas e precisas.")]),
            ("Análise", [
                ("Histograma", self.a_histograma, False, "Gráfico da distribuição de brilho/cores."),
                ("Equalizar", self.a_equalizar, False, "Melhora o contraste distribuindo melhor os tons."),
                ("Canais de cor", self.a_canais, False, "Separa azul, verde e vermelho e mostra estatísticas."),
                ("Binarizar", self.a_binarizar, False, "Preto e branco puro: limiar fixo, adaptativo ou Otsu."),
                ("Quantizar", self.a_quantizar, False, "Reduz a quantidade de níveis de cor.")]),
            ("Segmentar", [
                ("Crescer região", self.a_regioes, False, "Clique num ponto e a região de cor parecida é destacada."),
                ("Morfologia", self.a_morfologia, False, "Erosão, dilatação, abertura e fechamento."),
                ("Contornos", self.a_contornos, False, "Desenha o contorno dos objetos encontrados."),
                ("Linhas", self.a_linhas, False, "Encontra segmentos de reta."),
                ("Círculos", self.a_circulos, False, "Encontra círculos."),
                ("Cantos", self.a_cantos, False, "Marca cantos e pontos de destaque.")]),
            ("Faces", [
                ("Detectar faces", self.a_detectar, False, "Marca os rostos encontrados na imagem."),
                ("Reconhecer", self.a_reconhecer, False, "Identifica quem aparece na imagem."),
                ("Cadastrar aluno", self.a_cadastrar, True, "Guarda fotos de uma pessoa para o reconhecimento."),
                ("Treinar modelo", self.a_treinar, True, "Aprende os rostos cadastrados (refaça após cadastrar)."),
                ("Chamada (fotos)", self.a_chamada, True, "Faz a chamada a partir de foto, pasta de fotos ou vídeo."),
                ("Chamada (câmera)", self.a_chamada_camera, True, "Marca presença em tempo real pela câmera.")]),
        ]

    # ---------- infraestrutura ----------
    def _executar(self, fn, livre, host=None):
        if self._ao_vivo:
            self.status.set("Encerre a chamada ao vivo antes de usar outra ferramenta.")
            return
        if self._pedindo:
            self.status.set("Conclua ou cancele o ajuste em andamento antes de escolher outra ferramenta.")
            return
        if not livre and self.orig is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro (Ctrl+O).")
            return
        if host is not None:
            self._host = host
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            fn()
        except (cv2.error, ValueError, OSError, RuntimeError) as e:
            messagebox.showerror("Erro", str(e))
        finally:
            try:
                self.config(cursor="")
            except tk.TclError:  # janela fechada durante a operação
                pass

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
        self._titulo_res = f"Processada – {nome}"
        self.p_res.config(text=self._titulo_res)
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

    def pedir(self, titulo, campos, previa=None):
        """Mostra o formulário no painel de ajustes da aba atual e espera Aplicar/Cancelar.
        Se `previa(valores)` for dada, a imagem processada é atualizada a cada ajuste.
        Retorna o dicionário de valores, ou None se cancelado."""
        host = self._host
        for w in host.winfo_children():
            w.destroy()
        agendado = []

        def rodar_previa():
            agendado.clear()
            try:
                img = previa(form.valores())
            except (ValueError, cv2.error, OSError):
                self.status.set("Esse ajuste não pôde ser pré-visualizado.")
                return
            if img is not None:
                self.p_res.mostrar(img)
                self.p_res.config(text=f"Prévia – {titulo}")

        def mudou():
            if previa is None:
                return
            if agendado:
                self.after_cancel(agendado.pop())
            agendado.append(self.after(60, rodar_previa))

        form = Formulario(host, titulo, campos, ao_mudar=mudou)
        form.pack(fill="x")
        self._pedindo = True
        mudou()
        try:
            self.wait_variable(form.concluido)
        except tk.TclError:  # janela fechada
            return None
        finally:
            self._pedindo = False
            try:
                for a in agendado:
                    self.after_cancel(a)
                form.destroy()
                self._placeholder(host)
            except tk.TclError:
                pass
        if form.resultado is None and previa is not None:  # cancelado: volta ao que havia
            self.p_res.mostrar(self.res)
            self.p_res.config(text=self._titulo_res)
        return form.resultado

    def _ajustar(self, titulo, campos, calc, nome=None):
        """Padrão das ferramentas com ajuste: prévia ao vivo e, ao aplicar, entra no histórico.
        `calc(imagem_base, valores)` devolve a imagem resultante."""
        img = self.base()
        r = self.pedir(titulo, campos, previa=lambda v: calc(img, v))
        if r is not None:
            self.aplicar(calc(img, r), nome(r) if callable(nome) else (nome or titulo))
            return r
        return None

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
        self._titulo_res = "Processada"
        self.p_res.config(text=self._titulo_res)
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
        self._ajustar("Brilho e contraste", [
            ("c", "Contraste", float, 1.0, (0.2, 3.0, 0.05)),
            ("b", "Brilho", int, 0, (-255, 255, 1))],
            lambda im, r: p.brilho_contraste(im, r["c"], r["b"]), "Brilho/contraste")

    def a_saturacao(self):
        self._ajustar("Saturação", [("f", "Saturação", float, 1.0, (0.0, 3.0, 0.05))],
                      lambda im, r: p.saturacao(im, r["f"]))

    def a_somar(self):
        outra = self._abrir_imagem_extra()
        if outra is not None:
            self._ajustar("Somar imagens", [("w", "Peso da segunda imagem", float, 0.5, (0.0, 1.0, 0.05))],
                          lambda im, r: p.somar(im, outra, r["w"]), "Soma")

    def a_subtrair(self):
        img, outra = self.base(), self._abrir_imagem_extra()
        if outra is not None:
            self.aplicar(p.subtrair(img, outra), "Subtração |A−B|")

    def a_cinza(self):
        self.aplicar(p.cinza(self.base()), "Níveis de cinza")

    def a_hsv(self):
        def calc(im, r):
            m = r["m"]
            if m == "HSV → BGR":
                return p.hsv_para_bgr(im)
            if m == "Imagem HSV":
                return p.para_hsv(im)
            return cv2.split(p.para_hsv(im))["HSV".index(m[-1])]

        self._ajustar("RGB ⇄ HSV", [("m", "Exibir", "opcao", "Imagem HSV",
                                     ["Imagem HSV", "Canal H", "Canal S", "Canal V", "HSV → BGR"])],
                      calc, lambda r: r["m"])

    def a_redimensionar(self):
        def calc(im, r):
            if r["w"] and r["h"]:
                return p.redimensionar(im, r["w"], r["h"])
            return p.redimensionar(im, escala=r["e"])

        self._ajustar("Redimensionar", [
            ("e", "Escala", float, 1.0, (0.1, 3.0, 0.05)),
            ("w", "Largura px (0 = usar escala)", int, 0, (0, 4000, 10)),
            ("h", "Altura px", int, 0, (0, 4000, 10))], calc, "Redimensionada")

    def a_recortar(self):
        img = self.base()
        H, W = img.shape[:2]

        def calc(im, r):
            if r["mouse"]:
                return im
            return p.recortar(im, r["x"], r["y"], r["w"], r["h"])

        r = self._ajustar("Recortar região", [
            ("x", "X (esquerda)", int, 0, (0, W - 1, 1)),
            ("y", "Y (topo)", int, 0, (0, H - 1, 1)),
            ("w", "Largura", int, W, (1, W, 1)),
            ("h", "Altura", int, H, (1, H, 1)),
            ("mouse", "Em vez disso, selecionar com o mouse", bool, False)], calc, "ROI")
        if r and r["mouse"]:
            messagebox.showinfo("ROI", "Na janela que abrir, arraste o retângulo e tecle ENTER (C cancela).")
            x, y, w, h = p.selecionar_roi(img)
            self.desfazer()  # troca o resultado provisório (imagem inteira) pelo recorte
            self.aplicar(p.recortar(img, x, y, w, h), "ROI")

    def a_rotacionar(self):
        self._ajustar("Rotacionar", [
            ("a", "Ângulo (graus)", int, 0, (-180, 180, 1)),
            ("x", "Ampliar a tela para caber", bool, True)],
            lambda im, r: p.rotacionar(im, r["a"], r["x"]), lambda r: f"Rotação {r['a']}°")

    def a_espelhar(self):
        self._ajustar("Espelhar", [("m", "Direção", "opcao", "horizontal", ["horizontal", "vertical", "ambos"])],
                      lambda im, r: p.espelhar(im, r["m"]), lambda r: f"Espelho {r['m']}")

    # ---------- filtros ----------
    def a_media(self):
        self._ajustar("Filtro da média", [("k", "Tamanho do kernel", int, 5, (1, 31, 2))],
                      lambda im, r: p.filtro_media(im, r["k"]), "Média")

    def a_mediana(self):
        self._ajustar("Filtro da mediana", [("k", "Tamanho do kernel", int, 5, (1, 31, 2))],
                      lambda im, r: p.filtro_mediana(im, r["k"]), "Mediana")

    def a_gauss(self):
        self._ajustar("Filtro gaussiano", [("k", "Tamanho do kernel", int, 5, (1, 31, 2)),
                                           ("s", "Sigma (0 = automático)", float, 0.0, (0.0, 10.0, 0.1))],
                      lambda im, r: p.filtro_gaussiano(im, r["k"], r["s"]), "Gaussiano")

    def a_nitidez(self):
        self._ajustar("Nitidez", [("q", "Quantidade", float, 1.0, (0.0, 5.0, 0.1)),
                                  ("r", "Raio", int, 2, (1, 10, 1))],
                      lambda im, r: p.nitidez(im, r["q"], r["r"]), "Nitidez")

    def a_sobel(self):
        self._ajustar("Sobel", [("k", "Kernel", int, 3, (1, 7, 2))],
                      lambda im, r: p.sobel(im, r["k"]), "Sobel")

    def a_canny(self):
        self._ajustar("Canny", [("a", "Limiar inferior", int, 100, (0, 500, 1)),
                                ("b", "Limiar superior", int, 200, (0, 500, 1))],
                      lambda im, r: p.canny(im, r["a"], r["b"]), "Canny")

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
        self._ajustar("Binarização", [
            ("m", "Método", "opcao", "otsu", ["otsu", "fixo", "adaptativo"]),
            ("l", "Limiar (só no método fixo)", int, 127, (0, 255, 1))],
            lambda im, r: p.binarizar(im, r["m"], limiar=r["l"]), lambda r: f"Binarização ({r['m']})")

    def a_quantizar(self):
        self._ajustar("Quantização", [("n", "Níveis por canal", int, 4, (2, 64, 1))],
                      lambda im, r: p.quantizar(im, r["n"]), lambda r: f"Quantização ({r['n']} níveis)")

    # ---------- segmentação ----------
    def a_regioes(self):
        img = self.base()
        r = self.pedir("Crescimento de regiões", [("t", "Tolerância", int, 20, (0, 100, 1))])
        if not r:
            return
        messagebox.showinfo("Semente", "Na janela que abrir, clique no ponto inicial (ESC cancela).")
        semente = p.selecionar_semente(img)
        if semente is None:
            return
        mascara = p.crescimento_regioes(img, semente, r["t"])
        self.aplicar(p.destacar_regiao(img, mascara), f"Região ({int((mascara > 0).sum())} px)")

    def a_morfologia(self):
        def calc(im, r):
            base = p.binarizar(im, "otsu") if r["bin"] else im
            return p.morfologia(base, r["op"], r["k"], r["f"], r["i"])

        self._ajustar("Morfologia", [
            ("op", "Operação", "opcao", "erosao", ["erosao", "dilatacao", "abertura", "fechamento"]),
            ("k", "Kernel", int, 3, (1, 31, 2)),
            ("f", "Formato", "opcao", "retangulo", ["retangulo", "elipse", "cruz"]),
            ("i", "Iterações", int, 1, (1, 10, 1)),
            ("bin", "Binarizar (Otsu) antes", bool, True)], calc, lambda r: r["op"].capitalize())

    def _detector(self, titulo, campos, funcao, rotulo):
        """Ferramentas que desenham formas e informam quantas foram achadas."""
        n = []

        def calc(im, r):
            out, qtd = funcao(im, r)
            n[:] = [qtd]
            return out

        if self._ajustar(titulo, campos, calc, rotulo) is not None and n:
            self.status.set(f"{rotulo}: {n[0]} encontrado(s)")

    def a_contornos(self):
        self._detector("Contornos", [
            ("a", "Área mínima (px)", int, 100, (0, 5000, 10)),
            ("t1", "Limiar inferior", int, 50, (0, 255, 1)),
            ("t2", "Limiar superior", int, 150, (0, 255, 1))],
            lambda im, r: p.contornos(im, r["t1"], r["t2"], r["a"]), "Contornos")

    def a_linhas(self):
        self._detector("Linhas", [
            ("l", "Sensibilidade (menor = mais linhas)", int, 80, (10, 300, 5)),
            ("c", "Comprimento mínimo", int, 40, (5, 500, 5)),
            ("g", "Falha máxima", int, 10, (1, 100, 1))],
            lambda im, r: p.linhas(im, r["l"], r["c"], r["g"]), "Linhas")

    def a_circulos(self):
        self._detector("Círculos", [
            ("s", "Sensibilidade (menor = mais círculos)", int, 30, (10, 100, 1)),
            ("a", "Raio mínimo", int, 10, (0, 300, 1)),
            ("b", "Raio máximo (0 = livre)", int, 0, (0, 600, 1))],
            lambda im, r: p.circulos(im, r["s"], r["a"], r["b"]), "Círculos")

    def a_cantos(self):
        self._detector("Cantos", [
            ("n", "Máximo de cantos", int, 100, (1, 500, 1)),
            ("q", "Qualidade (%)", int, 1, (1, 50, 1))],
            lambda im, r: p.cantos(im, r["n"], r["q"] / 100), "Cantos")

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
            ("conf", "Reconhecimentos p/ marcar presença", int, 1, (1, 30, 1))])
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
        r = self.pedir("Chamada ao vivo", [
            ("idx", "Câmera (0 = primeira)", int, 0, (0, 5, 1)),
            ("conf", "Quadros de reconhecimento p/ marcar presença", int, 5, (1, 30, 1))])
        if not r:
            return
        sessao = rf.SessaoChamada(confirmacoes=r["conf"])
        cap = rf.abrir_camera(r["idx"])

        # a área do "Original" dá lugar à lista de presença; o vídeo anotado vai para a "Processada"
        self.p_orig.grid_remove()
        painel = ttk.LabelFrame(self.p_orig.master, text="Chamada ao vivo", padding=10)
        painel.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        info = tk.StringVar()
        ttk.Label(painel, textvariable=info, style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(painel, text="Presentes (na ordem em que foram reconhecidos):", foreground="#666"
                  ).pack(anchor="w", pady=(6, 2))
        lista = tk.Listbox(painel, font=("Segoe UI", 12), activestyle="none")
        lista.pack(fill="both", expand=True)
        ttk.Button(painel, text="⏹  Encerrar chamada", style="Acao.TButton",
                   command=lambda: self._encerrar_chamada()).pack(fill="x", pady=(10, 0))
        self.p_res.config(text="Câmera ao vivo")
        self._ao_vivo = True

        def encerrar():
            if not self._ao_vivo:
                return
            self._ao_vivo = None
            self._encerrar_chamada = lambda: None
            cap.release()
            painel.destroy()
            self.p_orig.grid()
            self.p_orig.mostrar(self.orig)
            self.p_res.mostrar(self.res)
            self.p_res.config(text=self._titulo_res)
            self.status.set(f"Chamada encerrada. Presentes: {', '.join(sorted(sessao.presentes)) or '-'}  |  "
                            f"Ausentes: {', '.join(sessao.ausentes) or '-'}. Registro em chamada.csv")

        self._encerrar_chamada = encerrar

        def passo():
            if not self._ao_vivo:
                return
            ok, quadro = cap.read()
            if ok:
                vis, novos = sessao.processar(quadro)
                self.p_res.mostrar(vis)
                for n in novos:
                    lista.insert("end", f"✔  {n}")
                info.set(f"Presentes: {len(sessao.presentes)} de {len(sessao.nomes)}")
                self.status.set("Chamada ao vivo em andamento…")
            self.after(15, passo)

        passo()


if __name__ == "__main__":
    App().mainloop()
