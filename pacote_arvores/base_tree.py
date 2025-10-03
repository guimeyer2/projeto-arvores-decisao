
import numpy as np
import pandas as pd
from .utils import most_common_label

class Node:
    """
    Nó da árvore de decisão.
    - Se for um nó de decisão, terá feature, threshold e children.
    - Se for um nó folha, terá apenas value.
    """
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature      # Atributo usado para a divisão
        self.threshold = threshold  # Limiar (contínuo) ou valor (categórico)
        self.children = children    # Dicionário de nós filhos
        self.value = value          # Valor da classe se for um nó folha

    def is_leaf_node(self):
        return self.value is not None

class BaseDecisionTree:
    """Classe base para as árvores de decisão."""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        # Garante que y seja um array numpy para bincount funcionar
        y = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 1. Condições de parada
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)

        # 2. Encontrar a melhor divisão (método a ser implementado por cada algoritmo)
        best_split = self._find_best_split(X, y)
        
        # Se não for encontrada uma divisão que melhore o critério, cria uma folha
        if best_split is None:
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)

        # 3. Criar os nós filhos recursivamente
        feature, threshold, child_indices = best_split
        
        children = {}
        for name, indices in child_indices.items():
            if len(indices) == 0:
                # Se um ramo não tiver amostras, cria uma folha com a classe mais comum do pai
                children[name] = Node(value=most_common_label(y))
            else:
                child_X, child_y = X.iloc[indices], y[indices]
                children[name] = self._grow_tree(child_X, child_y, depth + 1)

        return Node(feature, threshold, children)

    def _find_best_split(self, X, y):
        # Este método deve ser sobrescrito pelas classes filhas (ID3, C4.5, CART)
        raise NotImplementedError("Cada algoritmo deve implementar sua própria lógica de divisão.")

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_value = x[node.feature]
        
        # Para divisões binárias (C4.5 contínuo, CART)
        if isinstance(node.threshold, (int, float)):
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.children['<='])
            else:
                return self._traverse_tree(x, node.children['>'])
        # Para divisões categóricas (ID3, C4.5 categórico)
        else:
            # Se o valor visto na predição não existia no treino, não haverá um filho para ele.
            # Uma estratégia é retornar a classe mais comum do nó atual (requereria salvar essa info)
            # ou simplesmente parar (o que pode causar um erro).
            # Por simplicidade, assumimos que o valor existe.
            if feature_value in node.children:
                return self._traverse_tree(x, node.children[feature_value])
            else:
                # Estratégia de fallback: se um valor desconhecido aparece,
                # não podemos descer mais. Retornar um valor padrão é complexo.
                # Aqui, vamos quebrar, mas em produção seria necessário um tratamento melhor.
                # Para o projeto, isso é suficiente.
                # print(f"Aviso: valor '{feature_value}' para o atributo '{node.feature}' não encontrado na árvore.")
                return None # Ou uma classe padrão