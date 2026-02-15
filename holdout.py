import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional

def holdout(X,
            Y,
            test_size: float,
            random_state: int
            ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Divide il dataset in Training Set e Test Set in modo casuale ma riproducibile.
    """

    if test_size > 0.35:
        warnings.warn(
            "ATTENZIONE: stai utilizzando un training set < 65%. "
            "Questo potrebbe ridurre la capacità di apprendimento del modello.",
            UserWarning
        )

    # Imposta il seme per la generazione di numeri casuali per garantire 
    # che la divisione dei dati sia la stessa ad ogni esecuzione.
    np.random.seed(random_state)

    # Conta il numero totale di campioni presenti nel dataset.
    n_samples = len(X)

    # Crea un array di indici e li mescola casualmente.
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Calcola il punto di taglio di fine test set e inizio training set
    split_point = int(n_samples * (1 - test_size))

    # Suddivide gli indici mescolati in due gruppi: training set e test set.
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]

    # Estrazione dei dati: gestisce sia oggetti Pandas (DataFrame/Series) 
    # che semplici array Numpy.
    if hasattr(X, "iloc"):  # Se l'oggetto ha l'attributo .iloc, è di tipo Pandas
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]
    else:  # Altrimenti, viene trattato come un array Numpy standard
        X_train = X[train_idx]
        X_test = X[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

    # Restituisce i quattro set di dati convertiti in array Numpy 
    # per la compatibilità con gli algoritmi successivi.
    return (
        np.array(X_train),
        np.array(X_test),
        np.array(Y_train),
        np.array(Y_test)
    )