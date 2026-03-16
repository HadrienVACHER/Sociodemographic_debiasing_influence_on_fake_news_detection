from typing import Dict, List
import numpy as np
import scipy.linalg
from tqdm.auto import tqdm
import classifier

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) 
    return w_basis.dot(w_basis.T)

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int) -> np.ndarray:
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    return I - get_rowspace_projection(Q)

def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             min_accuracy: float, 
                             X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray,
                             X_train_main: np.ndarray = None, Y_train_main: np.ndarray = None, 
                             X_dev_main: np.ndarray = None, Y_dev_main: np.ndarray = None) -> tuple:
    
    # Copies des données de l'attribut (Jigsaw)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    
    # Copies des données de la tâche principale (LIAR) si elles sont fournies
    eval_main = (X_train_main is not None) and (Y_train_main is not None)
    if eval_main:
        X_train_main_cp = X_train_main.copy()
        X_dev_main_cp = X_dev_main.copy()

    rowspace_projections = []
    Ws = []
    P_intermediates = [np.eye(input_dim)]
    
    # NOUVEAU : Listes pour garder les deux accuracies en mémoire
    acc_attribute_list = []
    acc_main_list = []

    pbar = tqdm(range(num_classifiers), desc="INLP Iterations")
    for i in pbar:
        # 1. Évaluation Attribut Sensible (Jigsaw)
        clf_attr = classifier.SKlearnClassifier(classifier_class(**cls_params))
        acc_attr = clf_attr.train_network(X_train_cp, Y_train, X_dev_cp, Y_dev)
        acc_attribute_list.append(acc_attr)
        
        # 2. Évaluation Tâche Principale (LIAR)
        if eval_main:
            clf_main = classifier.SKlearnClassifier(classifier_class(**cls_params))
            acc_main = clf_main.train_network(X_train_main_cp, Y_train_main, X_dev_main_cp, Y_dev_main)
            acc_main_list.append(acc_main)
        
        pbar.set_description(f"iter: {i}, acc_attr: {acc_attr:.4f}")
        
        # Condition d'arrêt
        if acc_attr < min_accuracy: 
            print(f"\nArrêt anticipé : L'accuracy {acc_attr:.4f} est passée sous le seuil {min_accuracy}")
            break

        # Extraction de la direction et calcul de la projection
        W = clf_attr.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)

        P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
        P_intermediates.append(P)

        # On projette les DEUX jeux de données pour l'itération suivante
        X_train_cp = X_train.dot(P)
        X_dev_cp = X_dev.dot(P)
        if eval_main:
            X_train_main_cp = X_train_main.dot(P)
            X_dev_main_cp = X_dev_main.dot(P)

    P_final = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    # NOUVEAU : On retourne les listes d'accuracy à la fin !
    return P_final, rowspace_projections, Ws, P_intermediates, acc_attribute_list, acc_main_list