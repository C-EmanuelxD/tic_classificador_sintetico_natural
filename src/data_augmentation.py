import cv2
import numpy as np
import os
import random
import glob


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_RAIZ_DADOS = os.path.join(BASE_DIR, "..", "dataset")
PASTA_ORIGINAL = os.path.join(PASTA_RAIZ_DADOS, "original")
PASTA_AUMENTADA = os.path.join(PASTA_RAIZ_DADOS, "aumentado")
FATOR_AUMENTO = 10



def aplicar_rotacao_e_zoom(imagem):
    """Aplica rotações discretas (0/90/180/270) e um pequeno zoom/corte (Simulação Geométrica)."""
    
    angulo = random.choice([0, 90, 180, 270])
    if angulo == 90:
        imagem = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180:
        imagem = cv2.rotate(imagem, cv2.ROTATE_180)
    elif angulo == 270:
        imagem = cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = imagem.shape[:2]
    fator_escala = random.uniform(1.0, 1.2)
    novo_h, novo_w = int(h * fator_escala), int(w * fator_escala)
    
    imagem_escalada = cv2.resize(imagem, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)
    
    start_h = random.randint(0, novo_h - h)
    start_w = random.randint(0, novo_w - w)
    

    if novo_h >= h and novo_w >= w:
        imagem_cortada = imagem_escalada[start_h:start_h + h, start_w:start_w + w]
    else:

        return imagem
    
    return imagem_cortada

def aplicar_ruido_e_luminosidade(imagem):
    """Aplica variações de brilho/contraste e um ruído Gaussiano BEM SUAVE."""


    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-25, 25)

    imagem_saida = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)


    media = 0

    variancia = random.randint(1, 15)
    sigma = variancia ** 0.5
    

    gauss = np.random.normal(media, sigma, imagem_saida.shape).astype(np.float32)
    

    imagem_saida_float = imagem_saida.astype(np.float32)
    

    imagem_saida_float = imagem_saida_float + gauss
    

    imagem_saida = np.clip(imagem_saida_float, 0, 255).astype(np.uint8)

    return imagem_saida




def realizar_data_augmentation():
    

    classes = [d for d in os.listdir(PASTA_ORIGINAL) if os.path.isdir(os.path.join(PASTA_ORIGINAL, d))]
    total_imagens_geradas = 0

    print(f"Iniciando Data Augmentation para as classes: {classes}...")

    for classe in classes:
        
        caminho_classe_original = os.path.join(PASTA_ORIGINAL, classe)
        caminho_classe_aumentada = os.path.join(PASTA_AUMENTADA, classe)
        

        os.makedirs(caminho_classe_aumentada, exist_ok=True) 
        

        caminhos_imagens = glob.glob(os.path.join(caminho_classe_original, "*.jpg"))
        caminhos_imagens.extend(glob.glob(os.path.join(caminho_classe_original, "*.png")))
        
        print(f"  > Processando classe: {classe} ({len(caminhos_imagens)} imagens originais)")

        for caminho_original in caminhos_imagens:
            
            imagem_base = cv2.imread(caminho_original)
            if imagem_base is None:
                continue

            nome_arquivo_base = os.path.basename(caminho_original).split(".")[0]
            extensao = os.path.basename(caminho_original).split(".")[-1]
            

            nome_original_v0 = f"{nome_arquivo_base}_v0.{extensao}"
            cv2.imwrite(os.path.join(caminho_classe_aumentada, nome_original_v0), imagem_base)
            total_imagens_geradas += 1


            for i in range(1, FATOR_AUMENTO):
                
                imagem_aumentada = imagem_base.copy()
                

                imagem_aumentada = aplicar_ruido_e_luminosidade(imagem_aumentada)
                

                imagem_aumentada = aplicar_rotacao_e_zoom(imagem_aumentada)


                if random.choice([True, False]):
                    imagem_aumentada = cv2.flip(imagem_aumentada, 1)


                novo_nome = f"{nome_arquivo_base}_v{i}.{extensao}"
                caminho_novo = os.path.join(caminho_classe_aumentada, novo_nome)
                cv2.imwrite(caminho_novo, imagem_aumentada)
                total_imagens_geradas += 1

    print("-" * 50)
    print(f"✅ Data Augmentation concluído. Total de imagens geradas: {total_imagens_geradas}.")
    print("Rotulação mantida por pastas dentro de /datasets/aumentado/.")
    print("-" * 50)


if __name__ == "__main__":
    realizar_data_augmentation()