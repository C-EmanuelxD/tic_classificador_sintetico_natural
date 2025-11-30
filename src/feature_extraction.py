import cv2
import numpy as np
import os
import glob
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler

from segmentation import segmentar_gramado_por_cor, aplicar_mascara_na_imagem_cinza 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_AUMENTADA = os.path.join(BASE_DIR, "..", "dataset", "aumentado")
CAMINHO_FEATURES_SALVAS = os.path.join(BASE_DIR, "..", "dataset", "vetores_features.csv") 



def preprocessar_imagem(caminho_imagem):
    """Carrega, segmenta (isola o gramado), converte para cinza e normaliza os pixels."""
    
    imagem_colorida = cv2.imread(caminho_imagem)
    if imagem_colorida is None:
        return None


    _, mascara_gramado = segmentar_gramado_por_cor(imagem_colorida)


    imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
    

    imagem_cinza_mascarada = aplicar_mascara_na_imagem_cinza(imagem_cinza, mascara_gramado)
    

    imagem_normalizada = imagem_cinza_mascarada.astype(np.float32) / 255.0
    
    return imagem_normalizada



def extrair_LBP(imagem_normalizada):
    """Extrai o Histograma LBP (Local Binary Pattern) para textura estrutural."""
    
    imagem_uint8 = (imagem_normalizada * 255).astype(np.uint8)
    P = 8
    R = 1
    

    lbp_mapa = local_binary_pattern(imagem_uint8, P, R, method='uniform')
    
    n_bins = int(lbp_mapa.max() + 1)
    hist, _ = np.histogram(lbp_mapa.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    

    if len(hist) < 59:
        hist = np.pad(hist, (0, 59 - len(hist)), 'constant')
    elif len(hist) > 59:
        hist = hist[:59]

    return hist

def extrair_GLCM(imagem_normalizada):
    """Extrai as medidas estatísticas da Matriz de Co-ocorrência (GLCM) para textura estatística."""
    

    IMAGEM_QUANTIZADA = (imagem_normalizada * 31).astype(np.uint8) 
    
    ANGULOS = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 graus
    DISTANCIAS = [1]
    

    matriz_glcm = graycomatrix(IMAGEM_QUANTIZADA, 
                               DISTANCIAS, 
                               ANGULOS, 
                               levels=32, 
                               symmetric=True, 
                               normed=True)
    

    propriedades = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 
                    'contrast', 'ASM', 'entropy'] 
    
    vetor_GLCM_features = []
    
    for prop in propriedades:
 
        features = graycoprops(matriz_glcm, prop).ravel() 
        vetor_GLCM_features.extend(features)
        
    return np.array(vetor_GLCM_features)



def realizar_feature_extraction_e_fusao():
    
    CLASSES = {'sintetico': 0, 'natural': 1}
    dataset_features = [] 
    
    print(f"Iniciando a Extração de Características por subpasta de classe...")

    for nome_classe, rotulo in CLASSES.items():
        caminho_classe = os.path.join(PASTA_AUMENTADA, nome_classe)
        
        caminhos_imagens = glob.glob(os.path.join(caminho_classe, "*.jpg"))
        caminhos_imagens.extend(glob.glob(os.path.join(caminho_classe, "*.png"))) 

        print(f"  > Processando classe '{nome_classe}' (Rótulo: {rotulo}, {len(caminhos_imagens)} imagens)...")

        for i, caminho_imagem in enumerate(caminhos_imagens):
            
            imagem_normalizada = preprocessar_imagem(caminho_imagem)
            if imagem_normalizada is None:
                continue
                
            vetor_lbp = extrair_LBP(imagem_normalizada)
            vetor_glcm = extrair_GLCM(imagem_normalizada)
            

            vetor_final_features = np.concatenate([vetor_lbp, vetor_glcm])
            

            vetor_com_rotulo = np.append(vetor_final_features, rotulo)
            dataset_features.append(vetor_com_rotulo)

            if (i + 1) % 500 == 0:
                print(f"    {i + 1} imagens de '{nome_classe}' processadas...")




    
    dataset_features_array = np.array(dataset_features)
    
    if dataset_features_array.size == 0:
        print("-" * 50)
        print("ERRO: O conjunto de dados de features está VAZIO. Verifique se as imagens foram carregadas e processadas.")
        print("-" * 50)
        return
        

    if dataset_features_array.ndim == 1:

        DIMENSAO_FEATURE_TOTAL = 88 
        
        if dataset_features_array.size % DIMENSAO_FEATURE_TOTAL == 0:

            dataset_features_array = dataset_features_array.reshape(-1, DIMENSAO_FEATURE_TOTAL)
            print("AVISO: Array 1D corrigido para 2D via reshape.")
        else:
            print("ERRO: Array 1D tem tamanho inválido. O erro de indexação é inevitável. Abortando.")
            return
            

    X = dataset_features_array[:, :-1]
    y = dataset_features_array[:, -1]


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    dataset_final = np.hstack((X_scaled, y.reshape(-1, 1)))


    colunas_features = [f'F_{j+1}' for j in range(dataset_final.shape[1] - 1)]
    colunas_features.append('Rotulo')
    
    df = pd.DataFrame(dataset_final, columns=colunas_features)
    df['Rotulo'] = df['Rotulo'].astype(int) 
    
    df.to_csv(CAMINHO_FEATURES_SALVAS, index=False)
    
    print("-" * 50)
    print(f"✅ Extração de Características e Normalização concluídas.")
    print(f"Dataset final (Features Normalizadas) salvo em: {CAMINHO_FEATURES_SALVAS}")
    print(f"Total de amostras: {df.shape[0]}. Dimensão do vetor: {df.shape[1] - 1} (LBP 59 + GLCM 28 = 87).")
    print("-" * 50)

if __name__ == "__main__":
    realizar_feature_extraction_e_fusao()