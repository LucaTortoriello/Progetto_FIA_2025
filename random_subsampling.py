from holdout import holdout
from Data_Preprocessing import Data_Loader

def random_subsampling(X, Y, test_size=0.2, n=1, random_state=42):
    """
    Esegue il metodo di validazione Random Subsampling.
    Ripete la procedura di holdout 'n' volte per ottenere 
    una valutazione più robusta del modello.
    """
    
    # n = numero di iterazioni

    # Se n è uguale a 1, la tecnica coincide con un semplice holdout.
    # Viene chiamata direttamente la funzione holdout e restituito il singolo split.
    if n == 1:
        return holdout(X, Y, test_size=test_size, random_state=random_state)

    # Lista per memorizzare i risultati di ogni singola iterazione di holdout.
    results = [] 

    # Ciclo for per eseguire le 'n' ripetizioni richieste.
    for i in range(n):
        # Ad ogni iterazione dobbiamo cambiare il seed (seme).
        # Se usassimo sempre lo stesso 'random_state', otterremmo 'n' volte 
        # la stessa identica suddivisione dei dati.
        current_seed = random_state + i  # Il seed varia in modo deterministico ad ogni giro

        # Chiama la funzione holdout per generare uno split casuale basato sul seed corrente.
        split = holdout(
            X,
            Y,
            test_size=test_size,
            random_state=current_seed
        )

        # split tupla contenente: (X_train, X_test, Y_train, Y_test).
        # Se lo split è stato generato correttamente, viene aggiunto alla lista.
        if split is not None:
            results.append(split)

    # Restituisce la lista di tutti gli split generati.
    return results