# pylint: disable=no-member
import cv2

import mediapipe as mp
import numpy as np
import os

def process_landmarks(image_path, base_output_path, output_size=(256, 256)):
    # Inicializar o MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                      min_tracking_confidence=0.6, 
                                      min_detection_confidence=0.6)

    # Carregar a imagem
    image_bgr = cv2.imread(image_path)

    # Verificar se a imagem foi carregada corretamente
    if image_bgr is None:
        print(f"Erro: Imagem não encontrada ou falha ao carregar. Verifique o caminho do arquivo: {image_path}")
        return
    
    # Criar uma imagem com fundo preto do mesmo tamanho da original
    black_background = np.zeros_like(image_bgr)

    # Converter a imagem de BGR para RGB antes de processar
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Verificar se foram detectados landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image_rgb.shape[1])
                y = int(landmark.y * image_rgb.shape[0])
                landmark_points.append((x, y))
            
            # Plotar os pontos com uma linha fina (substitui o círculo)
            for point in landmark_points:
                cv2.line(black_background, point, point, (255, 255, 255), 1)  # Usando uma linha de espessura 1

        # Redimensionar a imagem processada para o tamanho desejado
        resized_black_background = cv2.resize(black_background, output_size, interpolation=cv2.INTER_LINEAR)

        # Extraindo o nome da emoção e o tipo (train, test, validation) do caminho original da imagem
        emotion_path = os.path.basename(os.path.dirname(image_path))  # Nome da pasta da emoção
        tipo = os.path.basename(os.path.dirname(os.path.dirname(image_path)))  # Nome da pasta de treino, teste, validacao

        # Construindo o novo caminho para salvar a imagem processada
        new_image_path = os.path.join(base_output_path, tipo, emotion_path, os.path.basename(image_path))

        # Criando o diretório, se necessário
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

        # Salvando a imagem processada com fundo preto e pontos preenchidos
        cv2.imwrite(new_image_path, resized_black_background)

        print(f"Imagem processada salva em: {new_image_path}")
    else:
        print(f"Landmarks não detectados para a imagem: {image_path}. Pulando...")

    # Limpar recursos
    del image_bgr, black_background, image_rgb
    if 'resized_black_background' in locals():
        del resized_black_background
    face_mesh.close()  # Fechar o face mesh para liberar recursos
    cv2.destroyAllWindows()

