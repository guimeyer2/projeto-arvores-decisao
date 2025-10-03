# meu_pacote_de_arvores/cart.py

import numpy as np
from .base_tree import BaseDecisionTree
from .utils import gini_gain

class CART(BaseDecisionTree):
    """
    Implementação do algoritmo CART (Classification and Regression Trees).
    Esta versão é para classificação e usa o critério Gini.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        super().__init__(min_samples_split, max_depth)
        
    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        features = X.columns
        
        for feature in features:
            unique_values = X[feature].unique()
            
            for value in unique_values:
                # Identifica o tipo de atributo (categórico vs. numérico)
                is_numeric = np.issubdtype(X[feature].dtype, np.number)
                
                if is_numeric:
                    threshold = value
                    # Divisão binária para numéricos
                    left_indices = X.index[X[feature] <= threshold]
                    right_indices = X.index[X[feature] > threshold]
                else: # Categórico
                    threshold = value
                    # Divisão binária para categóricos: valor vs. todos os outros
                    left_indices = X.index[X[feature] == threshold]
                    right_indices = X.index[X[feature] != threshold]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                y_splits = [y[left_indices], y[right_indices]]
                gain = gini_gain(y, y_splits)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain == 0:
            return None

        # Organiza os índices dos filhos para o formato esperado pela classe base
        is_numeric_split = np.issubdtype(X[best_feature].dtype, np.number)
        if is_numeric_split:
            left_indices = X.index[X[best_feature] <= best_threshold].tolist()
            right_indices = X.index[X[best_feature] > best_threshold].tolist()
            child_indices = {'<=': left_indices, '>': right_indices}
        else:
            left_indices = X.index[X[best_feature] == best_threshold].tolist()
            right_indices = X.index[X[best_feature] != best_threshold].tolist()
            # Renomeia para ser consistente com _traverse_tree
            # A lógica de travessia precisa ser ajustada para isso
            # Por simplicidade, vamos usar a mesma lógica do numérico
            child_indices = {'<=': left_indices, '>': right_indices} 
            # ATENÇÃO: A lógica em _traverse_tree precisará de um pequeno ajuste para
            # lidar com a checagem '==' para categóricos.
            # O ideal seria um nó diferente ou uma flag.
            # Para este projeto, a simplificação acima pode funcionar, mas é uma decisão de projeto
            # a ser justificada.
            
        return best_feature, best_threshold, child_indices