import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

def bootstrap(X: pd.DataFrame,
              Y: pd.DataFrame,
              k: int,
              random_state: int) -> Optional[List[Tuple[np.ndarray]]]:
    """
    Esegue la validazione tramite Bootstrap.
    Crea 'k' campionamenti casuali con reinserimento (replacement).
    """
    
    # Converte gli input in array NumPy per facilitare l'indicizzazione avanzata.
    X = np.array(X)
    Y = np.array(Y)
    n_samples = X.shape[0]  # Numero totale di campioni nel dataset originale.

    dataset_split = []
    
    # Genera 'k' diversi set di addestramento e test.
    for i in range(k):
        # Utilizza un seed variabile (random_state + i) per garantire che ogni 
        # campionamento sia diverso ma riproducibile.
        current_seed = random_state + i
        np.random.seed(current_seed)
        
        # Genera gli indici per il Training Set scegliendo 'n_samples' elementi con reinserimento.
        # Questo significa che alcuni campioni appariranno più volte nel training set.
        train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Mescola gli indici di training per aggiungere ulteriore variabilità.
        np.random.shuffle(train_indices)
        
        X_train = X[train_indices]
        Y_train = Y[train_indices]

        # Identifica i campioni "Out-of-Bag" (OOB), ovvero quelli mai estratti nel training set.
        # Questi formeranno il Test Set per questa iterazione.
        test_indices = list(set(range(n_samples)) - set(train_indices))
        X_test = X[test_indices]
        Y_test = Y[test_indices]

        # Aggiunge lo split corrente (4 array) alla lista dei risultati.
        dataset_split.append((X_train, X_test, Y_train, Y_test))

    # Controllo statistico: nel bootstrap standard, circa il 36.8% dei dati finisce nel test set.
    # Se la percentuale di campioni nel primo test set è inferiore al 30%, avvisa l'utente.
    if len(dataset_split[0][3]) / n_samples < 0.30:
        print("ATTENZIONE: stai utilizzando un test test_set troppo basso")
        risposta = input("Desideri continuare? (s/n): ").strip().lower()

        # Permette l'interruzione manuale se la distribuzione statistica non è soddisfacente.
        if risposta != 's':
            print("Esecuzione interrotta dall'utente.")
            return None

    # Restituisce la lista di tuple, coerente con il formato del Random Subsampling.
    return dataset_split