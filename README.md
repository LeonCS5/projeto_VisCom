# 👁️ VisCom – Sistema de Processamento de Imagens e Reconhecimento Facial

Software completo de Processamento Digital de Imagens (PDI), Visão Computacional e Reconhecimento Facial desenvolvido em Python e OpenCV. O sistema conta com duas interfaces: uma **Interface Gráfica (GUI)** intuitiva com visualização lado a lado e uma **Interface em Linha de Comando (CLI)** interativa via terminal.

---

## 🚀 Funcionalidades

### 🎨 Operações Básicas
- **Inversão de Cores:** Negativo da imagem (`bitwise_not`).
- **Brilho e Contraste:** Ajuste linear de escala e deslocamento com saturação em `[0, 255]`.
- **Operações Aritméticas:**
  - Soma ponderada / blending de duas imagens com redimensionamento automático.
  - Subtração absoluta ($|A - B|$) para detecção de diferenças.
- **Conversão de Espaços de Cor:**
  - BGR para Níveis de Cinza.
  - BGR para HSV e vice-versa, com visualização isolada dos canais $H$ (Matiz), $S$ (Saturação) e $V$ (Brilho).
- **Transformações Geométricas e Recorte:**
  - Redimensionamento por fator de escala ou dimensões fixas (largura/altura) com interpolação adequada (`INTER_AREA` / `INTER_CUBIC`).
  - Recorte de Região de Interesse (ROI) via seleção interativa com o mouse ou coordenadas manuais.

### 🔍 Filtragem e Detecção de Bordas
- **Suavização / Redução de Ruído:**
  - Filtro da Média (Box filter).
  - Filtro da Mediana (preservação de bordas e remoção de ruído sal e pimenta).
  - Filtro Gaussiano com desvio-padrão ($\sigma$) configurável.
- **Detecção de Bordas:**
  - Operador de Sobel (magnitude do gradiente $\sqrt{G_x^2 + G_y^2}$).
  - Detector de Canny com limiares inferior e superior ajustáveis.

### 📊 Histograma e Análise de Intensidade
- **Cálculo e Exibição de Histogramas:** Gráficos interativos por canal (Cinza ou BGR) utilizando Matplotlib.
- **Equalização de Histograma:** Equalização direta em tons de cinza ou equalização seletiva no canal de brilho ($V$) do espaço HSV para imagens coloridas.
- **Análise de Canais:** Visualização individual dos canais B, G e R com métricas estatísticas (média, desvio padrão, valores mínimo e máximo).
- **Binarização (Thresholding):**
  - Limiar Global Fixo.
  - Limiar Adaptativo (Gaussiano).
  - Limiar Automático de Otsu.
- **Quantização:** Redução da profundidade de cores em $N$ níveis de intensidade por canal.

### 🧩 Segmentação e Morfologia Matemática
- **Crescimento de Regiões:** Segmentação com vizinhança-8 a partir de semente interativa escolhida por clique do mouse, agrupando pixels por tolerância de cor/distância.
- **Destaque Visual:** Sobreposição translúcida verde sobre a área segmentada.
- **Operações Morfológicas:** Erosão, Dilatação, Abertura e Fechamento com elementos estruturantes em formato de Retângulo, Elipse ou Cruz, com controle de iterações e binarização prévia opcional.

### 👤 Reconhecimento Facial e Chamada Automática
- **Detecção Facial:** Classificador Haar Cascade pré-treinado (`haarcascade_frontalface_default.xml`).
- **Cadastro Guiado de Alunos:** Filtro que aceita fotos avulsas ou pastas, validando a presença de exatamente 1 rosto antes de salvar no banco de dados.
- **Treinamento LBPH:** Treinador `cv2.face.LBPHFaceRecognizer` com normalização de tamanho e equalização de histograma, gerando os artefatos `modelo_lbph.yml` e `rotulos.txt`.
- **Reconhecimento em Imagem:** Identificação visual com caixas delimitadoras e indicação da confiança/distância.
- **Sistema de Chamada Automática:**
  - Suporte a fotos de turma, diretório de fotos ou **vídeo** (`.mp4`, `.avi`, `.mov`, etc.).
  - Configuração de confirmações mínimas (frames consecutivos) para evitar falsos positivos.
  - Registro de presença com data e hora em `chamada.csv` (sem duplicatas no mesmo dia).
  - Salvamento de frames anotados com identificação dos alunos na pasta `chamada_anotada/`.

---

## 📁 Estrutura do Projeto

```text
projeto_VisCom/
├── gui.py                 # Interface gráfica em Tkinter + Pillow + Matplotlib
├── main.py                # Interface de linha de comando com menu interativo
├── processamento.py       # Funções puras de PDI e visão computacional
├── reconhecimento.py      # Pipeline de detecção, treino LBPH e chamada
├── requirements.txt       # Dependências do projeto
│
├── test_gui.py            # Testes automatizados da interface gráfica
├── test_menu.py           # Testes automatizados do menu em terminal
├── test_processamento.py  # Testes unitários das transformações de imagem
│
├── faces/                 # Banco de fotos dos alunos (faces/<aluno>/...)
├── chamada_anotada/       # Frames e fotos gerados pelo módulo de chamada
├── modelo_lbph.yml        # Modelo serializado do reconhecedor facial (gerado)
├── rotulos.txt            # Mapeamento ID -> Nome do aluno (gerado)
└── chamada.csv            # Planilha com histórico de presenças registradas
```

---

## 🛠️ Instalação e Pré-requisitos

Recomenda-se o uso de ambiente virtual Python (versão 3.9 ou superior).

1. **Clone ou acesse o repositório:**
   ```bash
   cd projeto_VisCom
   ```

2. **Crie e ative um ambiente virtual:**
   - **Linux / macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

> **Atenção:** O projeto utiliza `opencv-contrib-python<5` para incluir o módulo `cv2.face`, necessário para o reconhecedor LBPH.

---

## 🖥️ Como Executar

### 1. Interface Gráfica (Recomendado)
Interface visual moderna com abas, diálogos modais para parâmetros e painel duplo de comparação ("Original" vs "Processada"):
```bash
python gui.py
```
- **Modo Encadear:** Ao marcar a caixa *Encadear*, as operações seguintes são aplicadas sobre o último resultado obtido, permitindo encadear filtros e transformações.
- **Suporte a Acentos:** Tratamento de caminhos com caracteres especiais no Windows através de streams de bytes do NumPy.

### 2. Menu Interativo via Terminal (CLI)
Para ambientes headless ou preferência por linha de comando:
```bash
python main.py
```
- Exibe o menu numerado por categorias.
- Permite abrir imagens, aplicar filtros, encadear operações e inspecionar os resultados lado a lado via janela do OpenCV (`cv2.imshow`).

---

## 🧑‍🏫 Guia de Reconhecimento Facial e Chamada

1. **Cadastro dos Alunos:**
   - Pela GUI ou Terminal, utilize a opção **"Cadastrar aluno (fotos)"**.
   - Forneça o nome do aluno e uma pasta contendo suas fotos (ou uma foto individual).
   - O sistema valida automaticamente se a imagem possui exatamente 1 rosto nítido e a armazena em `faces/<nome_do_aluno>/`.
   - *Alternativa manual:* Crie diretamente a pasta `faces/<nome_do_aluno>/` e adicione imagens `.jpg` ou `.png`.

2. **Treinar o Modelo:**
   - Execute a opção **"Treinar modelo"**.
   - As fotos serão normalizadas para 200×200 pixels em escala de cinza e equalizadas.
   - Os arquivos `modelo_lbph.yml` e `rotulos.txt` serão gerados no diretório raiz.

3. **Realizar a Chamada:**
   - Selecione a opção **"Chamada automática"**.
   - Indique o arquivo de imagem, pasta com fotos da turma ou arquivo de vídeo.
   - Configure o número de confirmações necessárias (útil em vídeos para descartar falsos positivos ocasionais).
   - O sistema marcará os presentes, exibirá a listagem na tela e salvará o registro em `chamada.csv`, com as imagens anotadas em `chamada_anotada/`.

---

## 🧪 Execução dos Testes Automatizados

O projeto conta com uma suíte de testes com **46 testes automatizados** cobrindo todas as funções de processamento, a navegação de menus e a interface gráfica:

```bash
pytest
```

Para executar exibindo detalhes e relatórios verbosos:
```bash
pytest -v
```

---

## 📦 Tecnologias Utilizadas

- **[Python](https://www.python.org/):** Linguagem base do projeto.
- **[OpenCV (opencv-contrib-python)](https://opencv.org/):** Algoritmos de visão computacional, Haar Cascades e reconhecedor LBPH.
- **[NumPy](https://numpy.org/):** Manipulação vetorial e matricial de tensores de imagem.
- **[Matplotlib](https://matplotlib.org/):** Visualização e plotagem de histogramas.
- **[Pillow (PIL)](https://python-pillow.org/):** Ponte de conversão entre tensores OpenCV e o Canvas do Tkinter.
- **[Tkinter / ttk](https://docs.python.org/3/library/tkinter.html):** Framework de interface gráfica nativo.
- **[pytest](https://docs.pytest.org/):** Suíte de testes automatizados e integração contínua.
