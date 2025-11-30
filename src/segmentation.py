import cv2
import numpy as np

def segmentar_gramado_por_cor(imagem_colorida):
    """
    Segmenta o gramado usando limiares de cor no espaço HSV.
    Este método isola pixels dentro de um certo espectro verde.
    """
    


    hsv = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2HSV)



    


    limite_inferior_verde = np.array([30, 40, 40])
    limite_superior_verde = np.array([90, 255, 255])



    mascara = cv2.inRange(hsv, limite_inferior_verde, limite_superior_verde)
    


    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    




    imagem_segmentada = cv2.bitwise_and(imagem_colorida, imagem_colorida, mask=mascara)



    mascara_3ch = cv2.merge([mascara, mascara, mascara])
    
    return imagem_segmentada, mascara


def aplicar_mascara_na_imagem_cinza(imagem_cinza, mascara_1ch):
    """
    Aplica a máscara de segmentação de volta à imagem em tons de cinza
    para que apenas o gramado seja usado no cálculo do LBP/GLCM.
    """
    

    imagem_cinza_float = imagem_cinza.astype(np.float32)
    mascara_float = mascara_1ch.astype(np.float32) / 255.0
    


    imagem_mascarada_cinza = imagem_cinza_float * mascara_float
    
    return imagem_mascarada_cinza