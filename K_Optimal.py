import numpy as np
import pandas as pd
from typing import Tuple

from Model_Development import KNN_Classifier
from holdout import holdout
from MetricsEvaluation import MetricsEvaluator

class KNN_Optimal:
    """
    Questa classe è come un allenatore che fa fare dei provini al modello KNN
    con diversi valori di K per trovare il campione (il K migliore).
    """

    def __init__(self, X_train, Y_train, k_range, splits):
        """
        X_train: Le caratteristiche che il modello deve imparare.
        Y_train: Le risposte corrette.
        k_range: Una lista di numeri (es. da 1 a 20) che vogliamo testare come K.
        splits: Dati già divisi pronti per essere usati nei test.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.k_range = k_range
        self.results = {}  # Qui salveremo i risultati di ogni metodo
        self.splits = splits

    def K_Holdout(self, test_size, random_state) -> Tuple[int, float]:
        """
        METODO 1: HOLDOUT
        Prende i dati, ne mette una parte da parte (validation), allena il modello 
        sul resto e vede come va. Lo fa una volta sola per ogni K.
        """
        # Dividiamo i dati una sola volta: una parte per allenare, una per testare (valutare)
        X_sub_train, X_val, Y_sub_train, Y_val = holdout(
            self.X_train, self.Y_train, test_size, random_state
        )

        best_k = -1
        best_acc = 0.0

        # Proviamo ogni K nella nostra lista (es. prima K=1, poi K=2, ecc.)
        for k in self.k_range:
            knn = KNN_Classifier(K=k)
            knn.fit(X_sub_train, Y_sub_train) # Il modello impara

            prediction = knn.predict(X_val)    # Il modello prova a indovinare
            y_proba = knn.predict_proba(X_val) # Il modello dice quanto è sicuro

            # Chiediamo all'Evaluator: "Com'è andato questo K?"
            evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
            metriche = evaluator.get_metrics()
            acc = metriche['Accuracy']

            # Se questo K è meglio del precedente, me lo segno!
            if acc > best_acc:
                best_acc = acc
                best_k = k

        # Salviamo il vincitore
        self.results['Holdout'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc

    def K_random_subsampling(self, test_size, splits, random_state) -> Tuple[int, float]:
        """
        METODO 2: RANDOM SUBSAMPLING
        Invece di fidarsi di un solo test, ripete il test più volte su pezzi di dati 
        diversi e fa la media. È più affidabile del metodo 1.
        """
        k_mean_performances = {}

        for k in self.k_range:
            accuracies = [] # Qui metteremo i voti che K prende in ogni esame

            # Facciamo fare a K diversi esami (uno per ogni 'split')
            for split in splits:
                X_train_split, _, Y_train_split, _ = split
                
                # Dividiamo ulteriormente lo split per avere un piccolo set di validazione
                X_sub_train, X_val, Y_sub_train, Y_val = holdout(
                    X_train_split, Y_train_split, test_size, random_state
                )

                knn = KNN_Classifier(K=k)
                knn.fit(X_sub_train, Y_sub_train)
                prediction = knn.predict(X_val)
                y_proba = knn.predict_proba(X_val)

                evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
                acc = evaluator.get_metrics()['Accuracy']
                accuracies.append(acc) # Segniamo il voto di questo esame

            # Per questo K, facciamo la media di tutti i suoi voti
            k_mean_performances[k] = np.mean(accuracies)

        # Chi ha la media più alta vince!
        best_k = max(k_mean_performances, key=k_mean_performances.get)
        best_acc = k_mean_performances[best_k]

        self.results['Random Subsampling'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc

    def K_bootstrap(self, n_boot, random_state, splits, test_size) -> Tuple[int, float]:
        """
        METODO 3: BOOTSTRAP
        Simile al Random Subsampling, ma usa una tecnica di campionamento 
        particolare (spesso usata quando i dati sono pochi o si vuole estrema precisione).
        """
        k_performances = {}

        for k in self.k_range:
            accuracies = []

            for split in splits:
                # Nota: qui il codice riutilizza la logica di divisione holdout
                X_train_split, _, Y_train_split, _ = split

                X_sub_train, X_val, Y_sub_train, Y_val = holdout(
                    X_train_split, Y_train_split, test_size, random_state
                )

                knn = KNN_Classifier(K=k)
                knn.fit(X_sub_train, Y_sub_train)
                prediction = knn.predict(X_val)
                y_proba = knn.predict_proba(X_val)

                evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
                acc = evaluator.get_metrics()['Accuracy']
                accuracies.append(acc)

            # Media dei risultati per questo K
            avg_acc = np.mean(accuracies)
            k_performances[k] = avg_acc

        # Trova il K con la media migliore
        best_k = max(k_performances, key=k_performances.get)
        best_acc = k_performances[best_k]

        self.results['Bootstrap'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc