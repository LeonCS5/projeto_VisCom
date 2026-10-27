# 👁️ VisCom – Sistema de Processamento de Imagens e Reconhecimento Facial

Software completo de **Processamento Digital de Imagens (PDI)**, **Visão Computacional** e **Reconhecimento Facial com Chamada Automática**, desenvolvido em Python e OpenCV. O sistema conta com duas interfaces:
- **Interface Gráfica (GUI):** Interface desktop moderna em Tkinter + Pillow, com visualização lado a lado ("Original" vs "Processada"), formulários embutidos no painel lateral com **prévia em tempo real**, suporte a desfazer (Ctrl+Z) e 20 estilos artísticos prontos.
- **Interface em Linha de Comando (CLI):** Menu de terminal interativo para uso headless ou testes rápidos via OpenCV (`cv2.imshow`).

[![CI](https://github.com/LeonCS5/projeto_VisCom/actions/workflows/ci.yml/badge.svg)](https://github.com/LeonCS5/projeto_VisCom/actions/workflows/ci.yml)

## 🌐 Site do projeto

A página de apresentação estática do projeto está disponível em [leoncs5.github.io/projeto_VisCom](https://leoncs5.github.io/projeto_VisCom/).

---

## 🚦 Estado Atual do Projeto

O projeto encontra-se em **estado avançado, maduro e 100% funcional**, com todos os requisitos centrais implementados e validados por uma suíte completa de testes automatizados.

### Resumo dos Módulos

| Módulo / Recurso | Arquivo Principal | Status | Cobertura de Testes |
| :--- | :--- | :---: | :---: |
| **Operações Básicas** (brilho, contraste, soma, subtração, cinza, HSV, redimensionamento, recorte, rotação, espelhamento) | `processamento.py` | Concluído | 100% (`test_processamento.py`) |
| **Filtros e Realce** (média, mediana, gaussiano, nitidez, Sobel, Canny) | `processamento.py` | Concluído | 100% (`test_processamento.py`) |
| **Histograma e Intensidade** (cálculo, equalização em V/HSV, canais BGR, binarização fixa/adaptativa/Otsu, quantização) | `processamento.py` | Concluído | 100% (`test_processamento.py`) |
| **Segmentação e Formas** (crescimento de regiões, morfologia, contornos, retas Hough, círculos Hough, cantos Shi-Tomasi) | `processamento.py` | Concluído | 100% (`test_processamento.py`) |
| **Estilos Artísticos (20 Filtros tipo Instagram)** | `filtros.py` | Concluído | 100% (`test_filtros.py`) |
| **Pipeline Facial (Haar Cascade + LBPH)** | `reconhecimento.py` | Concluído | 100% (`test_processamento.py`) |
| **Chamada por Fotos, Pastas e Vídeos** | `reconhecimento.py` | Concluído | 100% (`test_processamento.py`) |
| **Chamada ao Vivo (câmera USB/webcam)** | `reconhecimento.py` | Concluído | 100% (`test_gui.py`, `test_menu.py`) |
| **Interface Gráfica com Ajustes Embutidos** | `gui.py` | Concluído | 100% (`test_gui.py`) |
| **Menu de Terminal (CLI)** | `main.py` | Concluído | 100% (`test_menu.py`) |
| **Suíte de Testes Automatizados** | `test_*.py` | **128 testes passando** | Execução via `pytest` |

> ℹ️ **Nota para quem acabou de clonar o repositório:**
> Arquivos como `modelo_lbph.yml`, `rotulos.txt`, `chamada.csv` e as fotos dentro de `faces/*` são dados gerados em tempo de execução e são intencionalmente ignorados pelo `.gitignore`. Após clonar, siga a seção [Primeiros Passos com Reconhecimento Facial](#-primeiros-passos-com-o-reconhecimento-facial) para gerar seu modelo local.

---

## 📁 Onde Está Cada Coisa (Arquitetura do Projeto)

O repositório é organizado de forma modular, com clara separação entre **lógica pura de processamento**, **regras biométricas**, **interfaces de usuário** e **testes**:

```text
projeto_VisCom/
│
├── 🧠 LÓGICA E PROCESSAMENTO
│   ├── processamento.py       # Funções puras de PDI e visão computacional (OpenCV + NumPy)
│   ├── filtros.py             # 20 estilos de imagem prontos compostos a partir do processamento
│   └── reconhecimento.py      # Detecção Haar Cascade, treino LBPH, chamada por foto/vídeo/câmera
│
├── 🖥️ INTERFACES DE USUÁRIO
│   ├── gui.py                 # Interface gráfica desktop (Tkinter + ttk + Pillow + Matplotlib)
│   └── main.py                # Interface de terminal (CLI) com menu textual numerado
│
├── 🧪 TESTES AUTOMATIZADOS
│   ├── test_processamento.py  # Testes unitários das transformações matemáticas de imagem
│   ├── test_filtros.py        # Testes de preservação de tipo/forma e invariantes dos filtros
│   ├── test_gui.py            # Testes da interface gráfica, formulários embutidos e fluxos
│   └── test_menu.py           # Testes dos fluxos e comandos do menu CLI
│
├── 📂 DADOS E ARTEFATOS LOCAIS (gerados em tempo de execução)
│   ├── faces/                 # Banco de fotos organizadas por pasta de aluno: faces/<nome>/...
│   │   └── README.md          # Instruções específicas sobre o banco de imagens
│   ├── chamada_anotada/       # Frames anotados salvos pelo processo de chamada
│   ├── modelo_lbph.yml        # Classificador facial LBPH treinado e serializado (gerado)
│   ├── rotulos.txt            # Mapeamento do ID numérico para o nome do aluno (gerado)
│   └── chamada.csv            # Planilha CSV com histórico de presenças registradas (gerado)
│
└── ⚙️ CONFIGURAÇÃO
    ├── requirements.txt       # Dependências do Python (opencv-contrib-python, numpy, pillow, etc.)
    ├── .gitignore             # Arquivos ignorados (fotos, modelos treinados, planilhas geradas)
    └── README.md              # Documentação geral do projeto
```

### Divisão de Responsabilidades

| Arquivo | O que deve ficar nele | O que NÃO deve ficar nele |
| :--- | :--- | :--- |
| **`processamento.py`** | Funções puras de imagem: recebem um `ndarray` BGR/Cinza e parâmetros, retornam um novo `ndarray`. Devem ser puras e não mutar o array original. | Diálogos com usuário, janelas, `print()`, chamadas de `input()`. |
| **`filtros.py`** | Composições estéticas de filtros inspirados no Instagram (Clarendon, Vintage, Noir, etc.) que utilizam as funções de `processamento.py`. | Lógica de interface com usuário ou gerenciamento de janelas. |
| **`reconhecimento.py`** | Algoritmos de detecção (`cv2.CascadeClassifier`), treino e predição (`cv2.face.LBPHFaceRecognizer`), sessão de chamada (`SessaoChamada`), leitura da câmera e gravação de CSV. | Telas Tkinter ou menus de texto interativos. |
| **`gui.py`** | Layouts Tkinter, abas, controles deslizantes (`Formulario`), manipulação de eventos do mouse/teclado, conversão OpenCV -> ImageTk e exibição em Canvas. | Cálculos matemáticos de filtros ou manipulação direta de matrizes de imagem sem usar os módulos de lógica. |
| **`main.py`** | Loop de menu textual no terminal, captura de entradas com `input()`, roteamento para as funções dos módulos e exibição rápida via `cv2.imshow`. | Redefinição de filtros já existentes em `processamento.py`. |

---

## 🛠️ Instalação e Configuração

### Pré-requisitos
- **Python 3.9 a 3.12** instalado no sistema.
- Câmera USB ou webcam (para a chamada ao vivo).

### 1. Clonar o repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd projeto_VisCom
```

### 2. Criar e ativar o ambiente virtual

- **No Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(Se houver erro de política de execução no PowerShell, rode antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **No Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

> ⚠️ **Atenção:** É mandatório utilizar `opencv-contrib-python<5` (já especificado no `requirements.txt`), pois o módulo `cv2.face` (necessário para o LBPH) não está presente na versão padrão do OpenCV.

---

## 🚀 Como Executar

### 1. Interface Gráfica (Recomendado)
A interface gráfica é ideal para uso diário, apresentações e demonstrações interativas:

```bash
python gui.py
```

**Principais recursos da GUI:**
- **Visualização Lado a Lado:** Painel esquerdo com a imagem "Original" e painel direito com o "Resultado".
- **Painel de Parâmetros Embutido:** Ao clicar em uma ferramenta, seus sliders e seletores abrem logo abaixo dos botões na própria aba (sem abrir pop-ups incômodos). A imagem processada atualiza em tempo real enquanto você ajusta as barras.
- **Botões do Painel:**
  - `✔ Aplicar`: Confirma a alteração e atualiza o histórico.
  - `↺`: Restaura os valores padrão da ferramenta.
  - `✖ Cancelar`: Descarta a prévia e volta ao resultado anterior.
- **Caixa "Encadear":** Quando marcada, a próxima operação é aplicada sobre o resultado anterior (permitindo combinar múltiplos filtros em sequência).
- **Desfazer:** Use o botão `Desfazer` ou o atalho `Ctrl+Z` (armazena até 30 passos).
- **Aba "Estilos":** 20 filtros pré-configurados com miniatura gerada a partir da própria foto e slider de intensidade (0% a 100%).
- **Zoom e Pan:** Suporte a arrastar a imagem e visualizar detalhes.

### 2. Interface via Terminal (CLI)
Para testes rápidos, depuração ou ambientes sem servidor gráfico:

```bash
python main.py
```

- Exibe menu numerado dividido em:
  1. Entrada / Saída
  2. Operações Básicas
  3. Filtragem e Realce
  4. Histograma e Intensidade
  5. Segmentação e Morfologia
  6. Reconhecimento Facial

---

## 🧑‍🎓 Primeiros Passos com o Reconhecimento Facial

O módulo biométrico é capaz de reconhecer alunos cadastrados e emitir relatórios de presença em tempo real ou a partir de mídias pré-gravadas.

### Passo 1: Cadastrar Alunos
O classificador precisa de fotos para aprender a reconhecer cada indivíduo:
- **Pela Interface:** Na aba **Faces**, clique em **Cadastrar aluno**. Digite o nome do aluno e selecione uma pasta com fotos dele (ou uma foto individual). O sistema valida se há exatamente 1 rosto nítido antes de salvar.
- **Manualmente:** Crie uma subpasta dentro de `faces/` com o nome da pessoa (ex.: `faces/maria/`) e cole ali fotos nos formatos `.jpg` ou `.png`.

### Passo 2: Treinar o Modelo
- Clique em **Treinar modelo** na aba **Faces** da GUI (ou opção correspondente no terminal).
- O algoritmo normaliza as faces (200x200 pixels em escala de cinza equalizada) e treina o `LBPHFaceRecognizer`.
- Serão gerados na raiz do projeto:
  - `modelo_lbph.yml`: Parâmetros do modelo treinado.
  - `rotulos.txt`: Mapeamento entre os IDs numéricos e os nomes dos alunos.

### Passo 3: Executar a Chamada

Há três maneiras de executar a chamada:
1. **Por Foto ou Pasta de Fotos:** Analisa as fotos da turma e marca quem estiver presente.
2. **Por Arquivo de Vídeo (`.mp4`, `.avi`, etc.):** Analisa os frames e marca presença após um número configurável de confirmações consecutivas.
3. **Pela Câmera ao Vivo:**
   - Escolha o número da câmera (0 = primeira; se houver webcam e câmera USB, tente 1).
   - Pressione `ESC` na janela para encerrar a sessão.

**Resultados Gerados:**
- `chamada.csv`: Planilha com data, horário e nome do aluno (evita registros duplicados no mesmo dia).
- `chamada_anotada/`: Pasta contendo cópia dos frames anotados com as caixas delimitadoras e nomes dos alunos reconhecidos.

---

## 🧪 Testes Automatizados

O projeto adota testes automatizados com `pytest` para garantir que nenhuma alteração quebre filtros ou componentes existentes.

Para rodar toda a suíte de testes:
```bash
pytest
```

Para rodar com detalhes de cada teste executado:
```bash
pytest -v
```

Para rodar testes de um arquivo específico:
```bash
pytest test_processamento.py -v
pytest test_filtros.py -v
pytest test_gui.py -v
pytest test_menu.py -v
```

---

## 🤝 Guia de Contribuição para o Grupo

Para que todos trabalhem em harmonia e sem conflitos de código, sigam estas diretrizes:

### 1. Fluxo de Trabalho com Git
- **Não trabalhe diretamente na branch `main`:** Crie uma branch para a sua funcionalidade:
  ```bash
  git checkout -b feature/nome-da-funcionalidade
  ```
- **Antes de abrir Pull Request / mergear:**
  1. Rode os testes e garanta que todos passem:
     ```bash
     pytest -v
     ```
  2. Atualize sua branch com a `main`:
     ```bash
     git checkout main
     git pull
     git checkout feature/nome-da-funcionalidade
     git merge main
     ```
- **Nunca force commit de fotos pesadas ou arquivos gerados:**
  Mantenha `faces/*`, `.yml`, `.csv` e pastas de cache ignorados pelo Git para não inflar o repositório.

### 2. Como Adicionar uma Nova Operação de PDI

Se você precisar criar uma nova operação de imagem:

1. **Implemente a função pura em `processamento.py`:**
   ```python
   def meu_novo_filtro(img, parametro=1.0):
       """Recebe imagem BGR e retorna nova imagem BGR processada."""
       saida = img.copy()
       # ... sua lógica com OpenCV ou NumPy ...
       return saida
   ```
2. **Adicione o teste unitário em `test_processamento.py`:**
   Garante que a função funciona, trata tipos adequadamente e não altera a imagem original.
3. **Exponha na Interface de Linha de Comando (`main.py`):**
   Crie a função de ação `a_meu_novo_filtro()` e adicione uma entrada na lista `MENU`.
4. **Exponha na Interface Gráfica (`gui.py`):**
   - Adicione o método `a_meu_novo_filtro(self)` na classe `App`.
   - Utilize `self._pedir_parametros()` com a lista de controles desejados (sliders, checkboxes ou comboboxes) para que os ajustes apareçam integrados ao painel lateral.
   - Adicione o botão no dicionário de abas correspondente (`Básicas`, `Realce`, `Análise` ou `Segmentar`).

### 3. Como Adicionar um Novo Estilo em `filtros.py`

1. Em `filtros.py`, crie a função com a assinatura:
   ```python
   def estilo_exemplo(img):
       # Utilize as funções de processamento ou OpenCV
       return resultado
   ```
2. Adicione a tupla `("Nome do Estilo", estilo_exemplo)` à lista `TODOS` em `filtros.py`.
3. A interface gráfica e os testes carregarão o novo estilo automaticamente!

---

## 💡 Ideias e Sugestões para Próximos Passos (Backlog)

Caso o grupo deseje expandir o projeto além dos requisitos básicos, aqui estão algumas sugestões com alto valor agregado:

- [ ] **Exportação de Relatórios de Chamada:** Gerar relatórios formatados em PDF ou Excel a partir do `chamada.csv`, com porcentagem de presença e gráficos.
- [ ] **Estatísticas da Turma:** Dashboard na GUI mostrando frequência por aluno e horários de entrada.
- [ ] **Feedback Sonoro na Chamada:** Emitir um bipe sonoro suave ao marcar a presença de um aluno pela câmera ao vivo.
- [ ] **Classificador Facial Alternativo (Deep Learning):** Adicionar opção de utilizar embeddings com modelos baseados em redes neurais (ex.: MediaPipe FaceMesh ou FaceNet) como alternativa ao LBPH em cenários de iluminação desafiadora.
- [ ] **Filtro de Olhos Fechados / Vivacidade (Liveness):** Evitar fraudes de chamada por foto estática usando detecção de piscadas.

---

## 📚 Tecnologias e Bibliotecas

- **[Python](https://www.python.org/)** (v3.9+) – Linguagem principal.
- **[OpenCV (opencv-contrib-python)](https://opencv.org/)** – Visão computacional, Haar Cascades e reconhecedor LBPH.
- **[NumPy](https://numpy.org/)** – Operações matriciais e manipulação de tensores de imagem.
- **[Pillow (PIL)](https://python-pillow.org/)** – Conversão de tensores OpenCV para exibição na interface gráfica.
- **[Matplotlib](https://matplotlib.org/)** – Gráficos de distribuição e plotagem de histogramas.
- **[Tkinter / ttk](https://docs.python.org/3/library/tkinter.html)** – Interface visual nativa multiplataforma.
- **[pytest](https://docs.pytest.org/)** – Framework de testes automatizados e integração contínua.
