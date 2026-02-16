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
        # Controllo di sicurezza: se non abbiamo le probabilità (scores), non possiamo variare la soglia
        if self.Y_scores is None:
            print("Errore: Servono gli 'scores' (probabilità) per fare la ROC!")
            return 0.0

        # --- 1. PREPARAZIONE DEI DATI ---
        
        # Ordiniamo gli indici in base ai punteggi in modo decrescente [::-1]
        # Vogliamo analizzare prima i campioni che il modello "crede" siano positivi con più forza
        desc_indices = np.argsort(self.Y_scores)[::-1]
        
        # Riapplichiamo l'ordine sia alle etichette reali che ai punteggi
        y_true_sorted = self.Y_true[desc_indices]
        y_scores_sorted = self.Y_scores[desc_indices]

        # Contiamo quanti positivi (classe 4) e negativi (classe 2) esistono in totale.
        # Questi saranno i nostri "valori massimi" per normalizzare gli assi tra 0 e 1.
        P_total = np.sum(self.Y_true == self.pos_label)
        N_total = np.sum(self.Y_true == self.neg_label)

        # Inizializziamo le liste delle coordinate (X=FPR, Y=TPR) partendo dall'origine (0,0)
        tpr_list = [0.0]
        fpr_list = [0.0]
        
        # Contatori cumulativi per i Veri Positivi e Falsi Positivi trovati durante la scansione
        tp_count = 0
        fp_count = 0

        # --- 2. SCANSIONE DEI CAMPIONI ---
        
        # Scorriamo tutti i campioni ordinati. In ogni iterazione è come se abbassassimo 
        # la soglia di classificazione per includere il campione corrente.
        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == self.pos_label:
                # Se il campione corrente è davvero positivo, incrementiamo il contatore Y
                tp_count += 1  
            else:
                # Se è un negativo (ma lo stiamo includendo come positivo), incrementiamo il contatore X
                fp_count += 1  

            # GESTIONE DEI PARI MERITO:
            # Se il prossimo campione ha lo stesso punteggio di quello attuale, non aggiungiamo 
            # ancora un punto al grafico. Aspettiamo di averli processati tutti per evitare
            # di creare scalini artificiali quando la soglia non è distinguibile.
            if i == len(y_scores_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                
                # Calcoliamo il True Positive Rate (Sensitivity): TP trovati / Totale Positivi
                tpr = tp_count / P_total if P_total > 0 else 0
                
                # Calcoliamo il False Positive Rate (1-Specificity): FP trovati / Totale Negativi
                fpr = fp_count / N_total if N_total > 0 else 0
                
                tpr_list.append(tpr)
                fpr_list.append(fpr)
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