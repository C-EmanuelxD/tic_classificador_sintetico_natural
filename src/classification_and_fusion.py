import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_FEATURES_SALVAS = os.path.join(BASE_DIR, "..", "dataset", "vetores_features.csv") 
PASTA_RESULTADOS = os.path.join(BASE_DIR, "..", "resultados")
os.makedirs(PASTA_RESULTADOS, exist_ok=True)



def calcular_metricas_e_matriz(y_true, y_pred, y_prob):
    """Calcula e retorna as métricas de desempenho para um fold ou resultado final."""
    
    macro_p = precision_score(y_true, y_pred, average='macro')
    macro_r = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    

    acc = accuracy_score(y_true, y_pred) 
    

    matriz = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': acc,
        'Macro_Precision': macro_p,
        'Macro_Recall': macro_r,
        'Macro_F1': macro_f1
    }, matriz




def regra_da_soma(probabilidades_c1, probabilidades_c2):
    """
    Aplica a regra da soma para fusão de decisão. 
    A classe escolhida é a que tiver a maior soma das predições de probabilidade.
    [cite: 1563-1566, 1579]
    """
    

    soma_probabilidades = probabilidades_c1 + probabilidades_c2
    

    y_pred_final = np.argmax(soma_probabilidades, axis=1)
    
    return y_pred_final



def realizar_classificacao_e_fusao():
    

    try:
        df = pd.read_csv(CAMINHO_FEATURES_SALVAS)
    except FileNotFoundError:
        print(f"ERRO: Arquivo {CAMINHO_FEATURES_SALVAS} não encontrado. Execute o 02_feature_extraction.py antes.")
        return


    X = df.drop('Rotulo', axis=1).values
    y = df['Rotulo'].values
    

    CLASSES_NOMES = {0: 'Sintetico', 1: 'Natural'}
    

    K_BEST = 5 
    modelo_knn = KNeighborsClassifier(n_neighbors=K_BEST) 


    modelo_svm = SVC(kernel='rbf', probability=True, random_state=42)
    

    N_FOLDS = 5
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    

    y_test_total = []
    y_pred_total_knn = []
    y_pred_total_svm = []
    y_pred_total_fusion = []
    
    print("-" * 50)
    print(f"Iniciando Classificação com {N_FOLDS}-Fold Cross-Validation...")

    for fold, (idx_treino, idx_teste) in enumerate(cv.split(X, y)):
        

        X_treino, X_teste = X[idx_treino], X[idx_teste]
        y_treino, y_teste = y[idx_treino], y[idx_teste]
        

        modelo_knn.fit(X_treino, y_treino)
        modelo_svm.fit(X_treino, y_treino)
        

        prob_knn = modelo_knn.predict_proba(X_teste)
        prob_svm = modelo_svm.predict_proba(X_teste)
        

        y_pred_fusion = regra_da_soma(prob_knn, prob_svm)
        

        y_pred_knn = modelo_knn.predict(X_teste)
        y_pred_svm = modelo_svm.predict(X_teste)
        

        y_test_total.extend(y_teste)
        y_pred_total_fusion.extend(y_pred_fusion)
        y_pred_total_knn.extend(y_pred_knn)
        y_pred_total_svm.extend(y_pred_svm)
        

        metricas_fold, _ = calcular_metricas_e_matriz(y_teste, y_pred_fusion, prob_knn)
        print(f"  > Fold {fold + 1} - Acurácia da Fusão: {metricas_fold['Accuracy']:.4f}")



    y_test_total = np.array(y_test_total)
    y_pred_total_fusion = np.array(y_pred_total_fusion)
    y_pred_total_knn = np.array(y_pred_total_knn)
    y_pred_total_svm = np.array(y_pred_total_svm)

    print("-" * 50)
    print("RESULTADOS FINAIS (Média da Validação Cruzada)")
    

    metricas_fusion, matriz_fusion = calcular_metricas_e_matriz(y_test_total, y_pred_total_fusion, None)
    print(f"1. FUSÃO (Regra da Soma):")
    print(f"   Acurácia: {metricas_fusion['Accuracy']:.4f}")
    print(f"   Macro F1: {metricas_fusion['Macro_F1']:.4f}")
    print(f"   Matriz de Confusão:\n{matriz_fusion}")
    

    metricas_knn, _ = calcular_metricas_e_matriz(y_test_total, y_pred_total_knn, None)
    metricas_svm, _ = calcular_metricas_e_matriz(y_test_total, y_pred_total_svm, None)
    
    print(f"2. k-NN (Individual): Acurácia {metricas_knn['Accuracy']:.4f} | Macro F1: {metricas_knn['Macro_F1']:.4f}")
    print(f"3. SVM (Individual): Acurácia {metricas_svm['Accuracy']:.4f} | Macro F1: {metricas_svm['Macro_F1']:.4f}")
    print("-" * 50)


    relatorio = pd.DataFrame([metricas_fusion, metricas_knn, metricas_svm],
                             index=['FUSÃO (Regra da Soma)', 'k-NN', 'SVM'])
    relatorio.to_csv(os.path.join(PASTA_RESULTADOS, 'relatorio_metricas_finais.csv'))
    print(f"✅ Relatório salvo em: {PASTA_RESULTADOS}/relatorio_metricas_finais.csv")

if __name__ == "__main__":
    realizar_classificacao_e_fusao()