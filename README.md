# Árvores de Decisão - ID3, C4.5 e CART

Implementação dos algoritmos de árvore de decisão **ID3**, **C4.5** e **CART** do zero para a disciplina de Inteligência Artificial.

## Características

- **ID3**: Ganho de informação, atributos categóricos
- **C4.5**: Razão de ganho, suporte a contínuos e missing values
- **CART**: Índice Gini, divisões sempre binárias

## Instalação

```bash
git clone https://github.com/guimeyer2/projeto-arvores-decisao.git
cd projeto-arvores-decisao
pip install -e .
```

## Uso

```python
from pacote_arvores import ID3, C45, CART
import pandas as pd

# Carregar dados
df = pd.read_csv('data/JogarTênis.csv')
X = df.drop('play', axis=1)
y = df['play']

# Treinar modelos
id3 = ID3()
id3.fit(X, y)

c45 = C45()
c45.fit(X, y)

cart = CART()
cart.fit(X, y)

# Fazer predições
predictions = id3.predict(X)
```

## Datasets

- **Play Tennis**: Dataset clássico (14 amostras, 4 features)
- **Titanic**: Dataset do Kaggle (891 amostras, 12 features)

## Estrutura do Projeto

```
├── pacote_arvores/          # Biblioteca principal
│   ├── __init__.py
│   ├── id3.py              # Algoritmo ID3
│   ├── c45.py              # Algoritmo C4.5
│   ├── cart.py             # Algoritmo CART
│   └── utils.py            # Funções utilitárias
├── data/                   # Datasets
├── relatorio.ipynb         # Análise completa
└── setup.py               # Configuração da biblioteca
```

## Resultados

| Algoritmo | Play Tennis | Titanic |
|-----------|-------------|---------|
| ID3       | 100%        | N/A*    |
| C4.5      | 100%        | 82%     |
| CART      | 93%         | 79%     |

*ID3 requer discretização para atributos contínuos

## Autor

Guilherme Meyer - Inteligência Artificial - 2025
