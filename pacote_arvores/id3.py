# meu_pacote_de_arvores/id3.py

import numpy as np
from .base_tree import BaseDecisionTree
from .utils import information_gain

class ID3(BaseDecisionTree):
    """
    Implementação do algoritmo ID3.
    Assume que todos os atributos de entrada são categóricos.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        super().__init__(min_samples_split, max_depth)

    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        
        features = X.columns
        
        for feature in features:
            unique_values = X[feature].unique()
            
            # Divide os rótulos y com base nos valores do atributo
            y_splits = [y[X[feature] == val] for val in unique_values]
            
            gain = information_gain(y, y_splits)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        if best_gain == 0:
            return None

        # Para ID3, o "threshold" é o próprio conjunto de valores
        # e os filhos são indexados por esses valores
        feature_values = X[best_feature].unique()
        child_indices = {val: X.index[X[best_feature] == val].tolist() for val in feature_values}

        return best_feature, feature_values, child_indices