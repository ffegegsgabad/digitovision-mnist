# DígitoVision — Reconhecimento de Dígitos Manuscritos (MNIST)

Projeto do **Mini-Projeto Avaliativo do Módulo 2** — curso *Desenvolvimento de IA para Análise
Preditiva [T1]* (SCTEC). Autor: **Rian Gomes**.

> **Vídeo de apresentação:** [Assistir no Google Drive](https://drive.google.com/drive/folders/1XKM58vIcCOOaypbfbcZDAOfIsaihAhOv?usp=sharing)

---

## Qual problema o sistema resolve?

**DígitoVision** é um pipeline de Machine Learning ponta a ponta que reconhece **dígitos
manuscritos (0 a 9)** a partir de imagens. Ele usa o dataset clássico de Visão Computacional
**MNIST** (70.000 imagens 28×28 em tons de cinza) para:

1. **Comparar 3 modelos** de aprendizado de máquina e descobrir qual reconhece melhor os dígitos;
2. **Testar a robustez** dos modelos diante de dados que eles **nunca viram** (cenário *Out-of-Distribution*);
3. **Reconhecer dígitos escritos à mão** pelo próprio usuário, a partir de uma foto.

É a base de aplicações reais como leitura automática de CEP, cheques, formulários e provas.

## Técnicas e tecnologias utilizadas

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.11 |
| Manipulação de dados | NumPy, Pandas |
| Visualização | Matplotlib, Seaborn |
| ML clássico | scikit-learn (Random Forest, KNN) |
| Deep Learning | TensorFlow / Keras (MLP) |
| Visão computacional | Pillow (PIL) |
| Ambiente | Jupyter Notebook |

**Conceitos aplicados:** análise exploratória (EDA), divisão estratificada treino/validação/teste,
normalização de pixels, ajuste de hiperparâmetros, matriz de confusão, métricas multiclasse
(*accuracy, precision, recall, F1*), *class masking*, inferência *Out-of-Distribution*, e um
pipeline de pré-processamento de imagens (escala de cinza, inversão, *bounding box*, centralização).

### Os 3 modelos comparados

| Modelo | Tipo | Hiperparâmetros ajustados |
|---|---|---|
| **Random Forest** | Clássico (ensemble de árvores) | `n_estimators=200`, `max_depth=20` |
| **KNN** | Clássico (baseado em distância) | `n_neighbors=3`, `weights='distance'` |
| **MLP** | Rede Neural (Keras) | arquitetura `128→64`, `learning_rate=0.001`, `epochs=15` |

## Resultados (resumo)

Desempenho no conjunto de teste (14.000 imagens):

| Modelo | Acurácia | Precisão | Recall | F1-Score | Tempo de treino |
|---|---|---|---|---|---|
| Random Forest | 96,52% | 96,53% | 96,52% | 96,52% | ~8 s |
| **KNN** | **97,21%** | **97,24%** | **97,21%** | **97,21%** | ~6 s |
| MLP (Rede Neural) | 97,14% | 97,16% | 97,14% | 97,13% | ~15 s |

Destaques do estudo:
- No *benchmark*, o **KNN teve a maior acurácia** (97,21%), seguido de perto pela MLP (97,14%) e pela
  Random Forest (96,52%).
- Para reconhecer **dígitos manuscritos próprios** (Fase 5.3), o projeto usa a **MLP**: mesmo com
  acurácia um pouco menor no teste, a rede neural **generaliza melhor** para caligrafias novas do que
  o KNN (que apenas memoriza o treino). É a diferença entre *"melhor no benchmark"* e *"melhor no
  mundo real"* — e o modelo reconhece corretamente o dígito manuscrito de exemplo.
- O **par de dígitos mais confundido** foi **4 e 9** (traços parecidos).
- O **Desafio B (OOD)** demonstra a **"falsa certeza"**: escondendo os dígitos **4 e 7** do treino,
  o modelo empurra a maioria (~64%) para o dígito **9** — e com **91,6% de confiança média**. Ou seja,
  **erra com convicção** diante de dados que nunca viu (risco real de IA em produção).

_Tabela completa em `outputs/tabela_comparativa.csv`; matrizes de confusão e gráficos em `outputs/graficos/`._

## Estrutura do projeto

```
digitovision-mnist/
├── notebook/
│   └── digitovision_mnist.ipynb   # notebook principal (todas as Fases)
├── data/
│   └── meus_digitos/              # imagens manuscritas próprias (Desafio C)
├── outputs/
│   ├── graficos/                  # figuras geradas (matrizes, curvas, grades)
│   └── modelos/                   # modelos treinados (.joblib / .keras)
├── requirements.txt
├── .gitignore
└── README.md
```

## Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/ffegegsgabad/digitovision-mnist.git
cd digitovision-mnist

# 2. Criar e ativar um ambiente virtual (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Abrir o notebook
jupyter notebook notebook/digitovision_mnist.ipynb
```

Depois, basta **executar as células na ordem** (menu *Run → Run All Cells*). O download do MNIST
(~15 MB) acontece automaticamente na Fase 1.

> **Testar com sua própria letra:** escreva dígitos em papel **branco** com caneta **escura**
> (ou desenhe no Paint com fundo **preto** e traço **branco**), salve as fotos em
> `data/meus_digitos/` e rode a Fase 5.3.

## Estrutura de branches (Git)

| Branch | Objetivo |
|---|---|
| `main` | Versão final e estável do projeto |
| `develop` | Integração das funcionalidades |
| `feat/eda` | Fase 1 — Análise Exploratória |
| `feat/preprocessamento` | Fase 2 — Divisão e normalização |
| `feat/modelos` | Fase 3 — Treinamento dos 3 modelos |
| `feat/avaliacao` | Fase 4 — Avaliação comparativa |
| `feat/robustez-ood` | Fase 5.1/5.2 — Class masking + OOD |
| `feat/imagens-proprias` | Fase 5.3 — Inferência com imagem própria |
| `docs/readme` | Documentação |

## Melhorias futuras

- Implementar uma **CNN** (rede convolucional) — "vê o contexto" e tende a superar a MLP em imagens;
- **Busca de hiperparâmetros** (GridSearch/RandomSearch) em vez de valores fixos;
- Ampliar o conjunto de imagens próprias e aplicar **data augmentation**;
- Adicionar **detecção de OOD** (rejeitar entradas de baixa confiança) para mitigar a falsa certeza;
- Disponibilizar o melhor modelo via **API (FastAPI)** com endpoint `/predict`.

---

_Projeto acadêmico — SCTEC / Módulo 2. Uso educacional._
