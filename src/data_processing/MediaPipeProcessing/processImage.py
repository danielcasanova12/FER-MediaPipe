import cv2
import mediapipe as mp
import numpy as np
import os

from .FullConected import process_landmarks_conected
from .essenciais import process_and_save_essenciais
from .filterDataSet import process_dataSet
from .landmarks import process_landmarks

def process_images_in_directory(base_directory, base_output_path, output_size=(512, 512), type='imagens_processed'):
    directories = ['treino', 'teste', 'validacao']
    for directory in directories:
        dir_path = os.path.join(base_directory, directory)
        
        if not os.path.exists(dir_path):
            print(f"Diretório {dir_path} não encontrado, pulando...")
            continue
        
        for class_dir in os.listdir(dir_path):
            class_path = os.path.join(dir_path, class_dir)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith('.png') or file.endswith('.jpg'):
                        image_path = os.path.join(class_path, file)
                        if type == 'imagens_processed':
                            process_dataSet(image_path, base_output_path)
                        elif type == 'essenciais':
                            process_and_save_essenciais(image_path, base_output_path)
                        elif type == 'landmarks':
                            process_landmarks(image_path, base_output_path, output_size)
                        elif type == 'landmarksConeted':
                            process_landmarks_conected(image_path, base_output_path)
                        else:
                            print('Tipo de processamento não encontrado')
                            break