import numpy as np
import matplotlib.pyplot as plt

class MetricsEvaluator:
    """
    Questa classe serve a misurare quanto è bravo un modello di Machine Learning.
    Prende i risultati reali e quelli predetti e calcola dei punteggi (metriche).
    """

    def __init__(self, Y_true, Y_pred, Y_scores=None, pos_label=4):        
        # Trasformiamo i dati in un formato "masticabile" da Python (Numpy array)
        # Se i dati arrivano da Pandas (Serie), usiamo .values, altrimenti np.array
        if hasattr(Y_true, 'values'):
            self.Y_true = Y_true.values
        else:
            self.Y_true = np.array(Y_true)

        if hasattr(Y_pred, 'values'):
            self.Y_pred = Y_pred.values
        else:
            self.Y_pred = np.array(Y_pred)

        # Gli "scores" sono le probabilità (es. "quanto è sicuro il modello che sia 4?")
        self.Y_scores = np.array(Y_scores) if Y_scores is not None else None

        # Definiamo chi è il "positivo" (es. il numero 4)
        self.pos_label = pos_label

        # Troviamo chi è il "negativo" (il 2)
        unique_classes = np.unique(self.Y_true)
        if len(unique_classes) == 2:
            # Se ci sono due classi, la negativa è quella che NON è pos_label
            self.neg_label = unique_classes[unique_classes != self.pos_label][0]
        else:
            # Caso di emergenza: se non le trova, usa 2 o 4 per default
            self.neg_label = 2 if self.pos_label == 4 else 4

        # Inizializziamo i contatori dei successi e dei fallimenti
        self.tp = self.tn = self.fp = self.fn = 0

    def get_metrics(self):
        """Calcola i 4 valori base: successi e tipi di errore"""
        
        # TP: Doveva essere 4 e il modello ha detto 4
        self.tp = np.sum((self.Y_true == self.pos_label) & (self.Y_pred == self.pos_label))
        # TN: Doveva essere 2 e il modello ha detto 2
        self.tn = np.sum((self.Y_true == self.neg_label) & (self.Y_pred == self.neg_label))
        # FP: Doveva essere 2 ma il modello ha detto 4 
        self.fp = np.sum((self.Y_true == self.neg_label) & (self.Y_pred == self.pos_label))
        # FN: Doveva essere 4 ma il modello ha detto 2
        self.fn = np.sum((self.Y_true == self.pos_label) & (self.Y_pred == self.neg_label))

    
        accuracy = (self.tp + self.tn) / len(self.Y_true) if len(self.Y_true) > 0 else 0
        
        
        error_rate = 1 - accuracy

        # Sensibilità (Recall): Su tutti i (4), quanti ne ha trovati?
        sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        # Specificità: Su tutti i (2), quanti ne ha riconosciuti come sani?
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        
        # Geometric Mean: media bilanciata tra Sensibilità e Specificità
        g_mean = np.sqrt(sensitivity * specificity)

        return {
            "Accuracy": round(float(accuracy * 100), 2),
            "Error Rate": round(float(error_rate * 100), 2),
            "Sensitivity": round(float(sensitivity * 100), 2),
            "Specificity": round(float(specificity * 100), 2),
            "Geometric Mean": round(float(g_mean * 100), 2)
        }

    def plot_confusion_matrix(self):
        """Crea un disegno (matrice) per vedere dove il modello si confonde"""
        self.get_metrics() # Aggiorna i calcoli

        # Prepariamo la tabella 2x2
        matrix_data = [[self.tn, self.fp], [self.fn, self.tp]]

        fig, ax = plt.subplots(figsize=(7, 7), facecolor='#f8f9fa')
        
        # Disegniamo i 4 quadrati
        for i in range(2):
            for j in range(2):
                val = matrix_data[i][j]
                # Se i == j (diagonale), sono corretti -> Blu. Altrimenti -> Rosso.
                bg_color = '#e7f3ff' if i == j else '#fff0f0'
                
                rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                    facecolor=bg_color, edgecolor='#ced4da', 
                                    linewidth=2)
                ax.add_patch(rect)
                
                # Scriviamo il numero grande al centro del quadrato
                ax.text(j, i, str(val), va='center', ha='center', 
                        fontsize=28, fontweight='bold', color='#2c3e50')

        # Settaggi grafici per rendere tutto bello e leggibile
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'PREDETTO\n{self.neg_label}', f'PREDETTO\n{self.pos_label}'], fontweight='bold')
        ax.set_yticklabels([f'REALE {self.neg_label}', f'REALE {self.pos_label}'], fontweight='bold')
        ax.xaxis.tick_top() # Sposta le scritte "Predetto" in alto
        plt.title('MATRICE DI CONFUSIONE', fontsize=16, fontweight='bold', pad=50)
        
    
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(1.6, -0.6)
        plt.show()

    def plot_roc_curve(self):
        """Disegna la curva ROC e calcola l'area (AUC)"""
        if self.Y_scores is None:
            print("Errore: Servono gli 'scores' (probabilità) per fare la ROC!")
            return 0.0

        # 1. Ordiniamo i campioni dal più "probabile 4" al meno probabile
        desc_indices = np.argsort(self.Y_scores)[::-1]
        y_true_sorted = self.Y_true[desc_indices]
        y_scores_sorted = self.Y_scores[desc_indices]


        # Quanti positivi (4) e negativi (2) ci sono in tutto?
        P_total = np.sum(self.Y_true == self.pos_label)
        N_total = np.sum(self.Y_true == self.neg_label)

        # Liste per salvare le coordinate (X, Y) del grafico
        tpr_list = [0.0]; fpr_list = [0.0]
        tp_count = 0; fp_count = 0

        # 2. Scendiamo lungo la lista (abbassando la soglia di decisione)
        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == self.pos_label:
                tp_count += 1  # Trovato un positivo -> la curva sale (Y)
            else:
                fp_count += 1  # Trovato un negativo -> la curva va a destra (X)

            # Se il prossimo ha lo stesso score, non aggiungere il punto ora
            if i == len(y_scores_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                tpr_list.append(tp_count / P_total if P_total > 0 else 0)
                fpr_list.append(fp_count / N_total if N_total > 0 else 0)

        # 3. Calcolo AUC (Area sotto la curva) con la regola del trapezio
        auc = 0.0
        for i in range(1, len(fpr_list)):
            larghezza = fpr_list[i] - fpr_list[i - 1]
            altezza_media = (tpr_list[i] + tpr_list[i - 1]) / 2
            auc += larghezza * altezza_media

        # 4. Disegno finale
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
        plt.fill_between(fpr_list, tpr_list, color='darkorange', alpha=0.1) # Colora l'area
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Linea del caso (50/50)
        plt.xlabel('Falsi Positivi (FPR)'); plt.ylabel('Veri Positivi (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.2)
        plt.show()

        return auc