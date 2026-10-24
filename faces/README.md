# Banco de Faces para Treinamento

Esta pasta armazena o banco de imagens dos alunos utilizado pelo modelo LBPH (`reconhecimento.py`).

## Estrutura Esperada

Organize as fotos criando uma subpasta para cada aluno com o nome desejado (sem espaços ou com underline):

```text
faces/
├── joao_silva/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── 0003.jpg
├── maria_souza/
│   ├── 0001.jpg
│   └── 0002.jpg
└── README.md
```

## Como Cadastrar Fotos

Você pode cadastrar fotos de duas formas:
1. **Pela Interface Gráfica ou Terminal (Recomendado):**
   - Utilize a opção **"Cadastrar aluno (fotos)"**.
   - O sistema valida automaticamente se a imagem possui exatamente 1 rosto antes de aceitar.
2. **Manualmente:**
   - Crie a subpasta com o nome do aluno dentro de `faces/`.
   - Adicione as fotos nos formatos `.jpg`, `.jpeg` ou `.png`.

> **Dica:** Quanto mais fotos com pequenas variações de ângulo, iluminação e expressão, melhor será a taxa de acerto do reconhecimento.
>
> Após adicionar novos alunos ou fotos, execute **"Treinar modelo"** para atualizar o `modelo_lbph.yml`.
