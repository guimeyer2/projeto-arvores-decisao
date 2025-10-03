# meu_pacote_de_arvores/c45.py

import numpy as np
from .base_tree import BaseDecisionTree
from .utils import gain_ratio

class C45(BaseDecisionTree):
    """
    Implementação (simplificada) do algoritmo C4.5.
    Lida com atributos categóricos e contínuos.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        super().__init__(min_samples_split, max_depth)
    
    def _find_best_split(self, X, y):
        best_gain_ratio = -1
        best_feature, best_threshold = None, None
        best_child_indices = None
        
        features = X.columns
        
        for feature in features:
            is_numeric = np.issubdtype(X[feature].dtype, np.number)
            
            if is_numeric:
                # Lógica para atributos contínuos (divisão binária)
                unique_values = sorted(X[feature].unique())
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values)-1)]
                
                for threshold in thresholds:
                    left_indices = X.index[X[feature] <= threshold]
                    right_indices = X.index[X[feature] > threshold]
                    
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue
                        
                    y_splits = [y[left_indices], y[right_indices]]
                    current_gain_ratio = gain_ratio(y, y_splits)
                    
                    if current_gain_ratio > best_gain_ratio:
                        best_gain_ratio = current_gain_ratio
                        best_feature = feature
                        best_threshold = threshold
                        child_indices = {'<=': left_indices.tolist(), '>': right_indices.tolist()}
                        best_child_indices = child_indices
            else:
                # Lógica para atributos categóricos (divisão multi-ramificada)
                unique_values = X[feature].unique()
                
                y_splits = [y[X[feature] == val] for val in unique_values]
                current_gain_ratio = gain_ratio(y, y_splits)
                
                if current_gain_ratio > best_gain_ratio:
                    best_gain_ratio = current_gain_ratio
                    best_feature = feature
                    best_threshold = unique_values # O "limiar" aqui é o conjunto de valores
                    child_indices = {val: X.index[X[feature] == val].tolist() for val in unique_values}
                    best_child_indices = child_indices

        if best_gain_ratio == 0:
            return None
            
        return best_feature, best_threshold, best_child_indices