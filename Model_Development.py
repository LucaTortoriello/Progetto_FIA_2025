import numpy as np
from collections import Counter
import random

# Implementazione del K-NN classificatore che ha come iper-parametro K
class KNN_Classifier:

    def __init__(self, K):
        # Inizializza il numero di vicini da considerare
        self.K = K

    def fit(self, X_train, Y_train): 
        # Salva i dati di addestramento. 
        self.X_train = np.array(X_train)
        # Assicura che le etichette siano in un formato monodimensionale.
        self.Y_train = np.array(Y_train).flatten()

    def predict(self, X_test): 
        # Converte l'input di test in array numpy per il calcolo delle distanze.
        X_test = np.array(X_test)
        predictions = []

        # Gestione del caso di un singolo campione di test
        if X_test.ndim == 1:
            X_test = [X_test]

        # Itera su ogni campione presente nel set di test
        for samples in X_test:
            # Calcola la distanza Euclidea tra il campione di test e tutti i campioni di training.
            distances = np.linalg.norm(self.X_train - samples, axis=1)

            # Associa ogni distanza alla corrispondente etichetta di classe.
            neighbors_data = list(zip(distances, self.Y_train))

            # Ordina le coppie in base alla distanza crescente.
            # In caso di distanze uguali, mantiene l'ordine originale di lettura.
            neighbors_data.sort(key=lambda x: x[0])

            # Seleziona i K vicini più prossimi.
            k_nearest = neighbors_data[:self.K]

            # Estrae solo le etichette di classe dei K vicini.
            k_labels = [label for (dist, label) in k_nearest]

            # Determina la classe più frequente tra i K vicini.
            conteggio = Counter(k_labels)
            max_common = conteggio.most_common(1)[0][1]
            
            # Gestione dei pareggi: se più classi hanno la stessa frequenza massima, 
            # le inserisce tutte nella lista best_labels.
            best_labels = [label for label, count in conteggio.items() if count == max_common]

            # Se c'è un pareggio, sceglie casualmente tra le classi migliori.
            predictions.append(random.choice(best_labels))

        # Restituisce l'array con le predizioni finali per ogni campione.
        return np.array(predictions)

    def predict_proba(self, X_test, pos_label=4):
        """
        Calcola la probabilità che i campioni di test appartengano alla classe positiva (pos_label).
        Utile per il calcolo della curva ROC e dell'AUC.
        """
        X_test = np.array(X_test)
        scores = []

        if X_test.ndim == 1:
            X_test = [X_test]

        for sample in X_test:
            # Calcola nuovamente le distanze Euclidee
            dists = np.linalg.norm(self.X_train - sample, axis=1)

            # Crea le coppie (distanza, etichetta) e le ordina per vicinanza.
            dist_label_pairs = zip(dists, self.Y_train)
            neighbors = sorted(dist_label_pairs, key=lambda x: x[0])[:self.K]

            # Estrae le etichette dei K vicini scelti.
            labels = [n[1] for n in neighbors]

            # Conta quante volte compare la classe definita come positiva (default 4).
            positive_votes = labels.count(pos_label)

            # La probabilità è la frequenza relativa della classe positiva tra i vicini.
            scores.append(positive_votes / self.K)

        # Restituisce un array di probabilità (valori tra 0.0 e 1.0).
        return np.array(scores)