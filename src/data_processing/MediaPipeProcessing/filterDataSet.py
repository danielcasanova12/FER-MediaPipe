import os
import cv2
import numpy as np
import mediapipe as mp
import itertools
import matplotlib.pyplot as plt

def process_dataSet(image_path, base_path):
    try:
        # Inicializando FaceMesh e definições
        mp_face_mesh = mp.solutions.face_mesh
        LEFT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
        RIGHT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
        LEFT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
        RIGHT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
        LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
        OTHER = [1]


        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        ) as face_mesh:

            # Carregando e processando a imagem
            img = cv2.imread(image_path)
            img = cv2.resize(img, (300, 300))
            img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
            img_shape = img.shape[0]

            results = face_mesh.process(img)


           

            # Extraindo o nome da emoção e o tipo (train, test, validation) do caminho original da imagem
            emotion_path = os.path.basename(os.path.dirname(image_path))  # Nome da pasta da emoção
            tipo = os.path.basename(os.path.dirname(os.path.dirname(image_path)))  # Nome da pasta de train, test, validation

            # Construindo o novo caminho para salvar a imagem processada
            new_image_path = os.path.join(base_path, tipo, emotion_path, os.path.basename(image_path))

            # Criando o diretório, se necessário
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)

            # Salvando a imagem processada
            cv2.imwrite(new_image_path, img)

            print(f"Imagem salva em: {new_image_path}")

    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")