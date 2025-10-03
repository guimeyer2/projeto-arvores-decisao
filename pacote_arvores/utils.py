import numpy as np
from collections import Counter

# --- Funções de Impureza ---

def entropy(y):
    """Calcula a entropia de um conjunto de rótulos."""
    # Conta a ocorrência de cada classe
    hist = np.bincount(y)
    ps = hist / len(y)
    # Retorna o cálculo da entropia. Adiciona-se um valor pequeno para evitar log(0).
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gini_index(y):
    """Calcula o índice Gini de um conjunto de rótulos."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum(ps**2)

# --- Funções para Critério de Divisão ---

def information_gain(y, y_splits):
    """
    Calcula o Ganho de Informação.
    y: rótulos do nó pai.
    y_splits: lista de arrays, cada um contendo os rótulos de um nó filho.
    """
    parent_entropy = entropy(y)
    n_total = len(y)
    
    # Calcula a entropia ponderada dos filhos
    weighted_child_entropy = sum((len(y_child) / n_total) * entropy(y_child) for y_child in y_splits)
    
    return parent_entropy - weighted_child_entropy

def gain_ratio(y, y_splits):
    """
    Calcula a Razão de Ganho.
    """
    ig = information_gain(y, y_splits)
    n_total = len(y)
    
    # Calcula o Split Information
    split_info = -np.sum([(len(y_child) / n_total) * np.log2(len(y_child) / n_total) for y_child in y_splits if len(y_child) > 0])
    
    # Evita divisão por zero
    if split_info == 0:
        return 0
        
    return ig / split_info

def gini_gain(y, y_splits):
    """
    Calcula a redução na impureza Gini.
    """
    parent_gini = gini_index(y)
    n_total = len(y)
    
    # Calcula o Gini ponderado dos filhos
    weighted_child_gini = sum((len(y_child) / n_total) * gini_index(y_child) for y_child in y_splits)
    
    # O "ganho" é a redução da impureza
    return parent_gini - weighted_child_gini

# --- Funções Auxiliares ---
def most_common_label(y):
    """Retorna o rótulo mais comum em um conjunto."""
    counter = Counter(y)
    if not counter:
        return None
    # Em caso de empate, retorna o primeiro encontrado
    return counter.most_common(1)[0][0]

def find_best_threshold(X_feature, y):
    """
    Encontra o melhor limiar para divisão binária em atributos contínuos.
    Retorna uma lista de limiares candidatos (pontos médios entre valores únicos).
    """
    unique_values = sorted(X_feature.unique())
    if len(unique_values) <= 1:
        return []
    
    thresholds = []
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        thresholds.append(threshold)
    
    return thresholds

def handle_missing_values(X, strategy='mode'):
    """
    Trata valores ausentes nos dados.
    strategy: 'mode' para categóricos, 'mean' para numéricos
    """
    X_filled = X.copy()
    
    for column in X_filled.columns:
        if X_filled[column].isnull().any():
            if np.issubdtype(X_filled[column].dtype, np.number):
                # Para numéricos, usa a média
                X_filled[column].fillna(X_filled[column].mean(), inplace=True)
            else:
                # Para categóricos, usa a moda
                mode_value = X_filled[column].mode()
                if len(mode_value) > 0:
                    X_filled[column].fillna(mode_value[0], inplace=True)
    
    return X_filled