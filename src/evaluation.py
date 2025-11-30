import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_RESULTADOS = os.path.join(BASE_DIR, "..", "resultados")
CAMINHO_FEATURES_SALVAS = os.path.join(BASE_DIR, "..", "dataset", "vetores_features.csv") 

CLASSES_NOMES = ['Sintetico', 'Natural']
N_FOLDS = 5



def regra_da_soma(probabilidades_c1, probabilidades_c2):
    """
    Aplica a regra da soma para fusão de decisão. [cite: 1904]
    A classe escolhida é a que tiver a maior soma das predições de probabilidade.
    """
    soma_probabilidades = probabilidades_c1 + probabilidades_c2
    y_pred_final = np.argmax(soma_probabilidades, axis=1)
    return y_pred_final



def plotar_matriz_confusao(matriz, titulo, caminho_salvar):
    """Gera e salva a matriz de confusão visualmente."""
    plt.figure(figsize=(8, 6))

    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES_NOMES, yticklabels=CLASSES_NOMES)
    plt.title(titulo)
    plt.ylabel('Classe Real (Actual Class)')
    plt.xlabel('Classe Predita (Predicted Class)')
    plt.savefig(caminho_salvar)
    plt.close()
    print(f"  > Matriz salva em: {caminho_salvar}")


def realizar_avaliacao_final():
    

    try:
        df = pd.read_csv(CAMINHO_FEATURES_SALVAS)
    except FileNotFoundError:
        print(f"ERRO: Arquivo {CAMINHO_FEATURES_SALVAS} não encontrado. Execute o 02_feature_extraction.py antes.")
        return

    X = df.drop('Rotulo', axis=1).values
    y = df['Rotulo'].values
    


    modelo_knn = KNeighborsClassifier(n_neighbors=5) 
    modelo_svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    

    y_test_total = []
    y_pred_fusion_total = []
    

    y_prob_fusion_total = np.zeros((len(X), len(CLASSES_NOMES)))
    
    print("-" * 50)
    print(f"Replicando {N_FOLDS}-Fold CV para obter as predições completas...")
    

    for fold, (idx_treino, idx_teste) in enumerate(cv.split(X, y)):
        
        X_treino, X_teste = X[idx_treino], X[idx_teste]
        y_teste = y[idx_teste]
        

        modelo_knn.fit(X_treino, y[idx_treino])
        modelo_svm.fit(X_treino, y[idx_treino])
        
        prob_knn = modelo_knn.predict_proba(X_teste)
        prob_svm = modelo_svm.predict_proba(X_teste)
        

        y_pred_fusion = regra_da_soma(prob_knn, prob_svm)
        

        y_test_total.extend(y_teste)
        y_pred_fusion_total.extend(y_pred_fusion)

    y_test_total = np.array(y_test_total)
    y_pred_fusion_total = np.array(y_pred_fusion_total)


    

    matriz_fusion = confusion_matrix(y_test_total, y_pred_fusion_total)
    plotar_matriz_confusao(matriz_fusion, 
                           "Matriz de Confusão - Fusão LBP+GLCM (SVM + k-NN)",
                           os.path.join(PASTA_RESULTADOS, 'matriz_fusao_final.png'))
    


    relatorio_classificacao = classification_report(y_test_total, y_pred_fusion_total, 
                                                    target_names=CLASSES_NOMES, output_dict=True)
    

    df_relatorio = pd.DataFrame(relatorio_classificacao).transpose()
    df_relatorio.to_csv(os.path.join(PASTA_RESULTADOS, 'relatorio_detalhado_classificacao.csv'))
    
    print("\n" + "=" * 60)
    print("RELATÓRIO DE DESEMPENHO FINAL (FUSÃO REGRA DA SOMA)")
    print("=" * 60)
    print(classification_report(y_test_total, y_pred_fusion_total, target_names=CLASSES_NOMES))
    
    print("-" * 60)
    print("MÉTRICAS CHAVE PARA O ARTIGO:")

    print(f"MACRO AVG F1-Score: {df_relatorio.loc['macro avg', 'f1-score']:.4f}")

    print(f"WEIGHTED AVG F1-Score: {df_relatorio.loc['weighted avg', 'f1-score']:.4f}")
    print(f"Acurácia Geral: {df_relatorio.loc['accuracy', 'support']:.4f}")
    print("Matriz de Confusão (Sintetico/Natural):\n", matriz_fusion)
    print("-" * 60)
    

if __name__ == "__main__":
    realizar_avaliacao_final()