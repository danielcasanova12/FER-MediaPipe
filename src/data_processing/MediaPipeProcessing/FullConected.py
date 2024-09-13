import os
import cv2
import mediapipe as mp
import numpy as np

def process_landmarks_conected(image_path, output_path):
    # Inicializar o MediaPipe FaceMesh com refine_landmarks=True para incluir os landmarks da íris
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                      refine_landmarks=True,  # Adiciona os landmarks da íris
                                      min_tracking_confidence=0.65, 
                                      min_detection_confidence=0.65)

    # Carregar a imagem
    image_bgr = cv2.imread(image_path)

    # Verificar se a imagem foi carregada corretamente
    if image_bgr is None:
        print(f"Erro: Imagem não encontrada ou falha ao carregar. Verifique o caminho do arquivo: {image_path}")
        return
    
    # Converter a imagem de BGR para RGB antes de processar
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Criar uma imagem preta para o fundo
    black_background = np.zeros_like(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image_rgb.shape[1])
                y = int(landmark.y * image_rgb.shape[0])
                landmark_points.append((x, y))
            
            # Usar todas as conexões disponíveis para a malha facial
            face_mesh_connections = [
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_face_mesh.FACEMESH_IRISES,
            ]
            
            for connections in face_mesh_connections:
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]

                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        start_point = landmark_points[start_idx]
                        end_point = landmark_points[end_idx]

                        cv2.line(black_background, start_point, end_point, (255, 255, 255), 1)  # Desenhar a linha branca sobre o fundo preto

            # Destacar a íris do olho esquerdo e direito em tons de cinza
            for idx in range(468, 473):
                iris_point = landmark_points[idx]
                cv2.circle(black_background, iris_point, 1, (192, 192, 192), -1)  # Desenhar um círculo cinza claro para a íris

            # Conectar os pontos da íris do olho esquerdo
            iris_connections_left = [(469, 470), (470, 471), (471, 472), (472, 469)]
            for connection in iris_connections_left:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv2.line(black_background, start_point, end_point, (192, 192, 192), 2)  # Conectar com uma linha cinza claro

            # Destacar a íris do olho direito em tons de cinza
            for idx in range(473, 478):
                iris_point = landmark_points[idx]
                cv2.circle(black_background, iris_point, 1, (192, 192, 192), -1)  # Desenhar um círculo cinza claro para a íris

            # Conectar os pontos da íris do olho direito
            iris_connections_right = [(474, 475), (475, 476), (476, 477), (477, 474)]
            for connection in iris_connections_right:
                start_idx, end_idx = connection
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    start_point = landmark_points[start_idx]
                    end_point = landmark_points[end_idx]
                    cv2.line(black_background, start_point, end_point, (192, 192, 192), 2)  # Conectar com uma linha cinza claro

    # Salvar a imagem com o fundo preto
    cv2.imwrite(output_path, cv2.cvtColor(black_background, cv2.COLOR_RGB2BGR))