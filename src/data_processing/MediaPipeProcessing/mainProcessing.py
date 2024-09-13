
from src.data_processing.MediaPipeProcessing.processImage import process_images_in_directory


dir_affectnet = dataset_path = r'F:\Git\Teste\FER\affectnet\affectnet2'
dir_affectnet_imagens_processed = r'F:\Git\Teste\FER-MediaPipe\data\affectnet\imagens_processed'
dir_affectnet_essenciais = r'F:\Git\Teste\FER-MediaPipe\data\affectnet\essenciais'
dir_affectnet_landmarks = r'F:\Git\Teste\FER-MediaPipe\data\affectnet\landmarks'
dir_affectnet_landmarksConeted = r'F:\Git\Teste\FER-MediaPipe\data\affectnet\landmarksConected'


dir_fer = 'C:/Users/leona/OneDrive/Documentos/Insper/9º Semestre/TCC2/Emotion_Recognition/data/fer2013'
dir_fer_imagens_processed = 'C:/Users/leona/OneDrive/Documentos/Insper/9º Semestre/TCC2/Emotion_Recognition/data/affectnet2'
dir_fer_essenciais = 'C:/Users/leona/OneDrive/Documentos/Insper/9º Semestre/TCC2/Emotion_Recognition/data/affectnet_Essenciais'
dir_fer_landmarks = 'C:/Users/leona/OneDrive/Documentos/Insper/9º Semestre/TCC2/Emotion_Recognition/data/affectnet_landmarks'
dir_fer_landmarksConeted = 'C:/Users/leona/OneDrive/Documentos/Insper/9º Semestre/TCC2/Emotion_Recognition/data/affectnet_landmarks2'




# Processar e salvar as imagens com o novo tamanho
process_images_in_directory(dir_affectnet, dir_affectnet_imagens_processed, output_size=(512, 512), type='imagens_processed')
process_images_in_directory(dir_affectnet, dir_affectnet_essenciais, output_size=(512, 512), type='essenciais')
process_images_in_directory(dir_affectnet, dir_affectnet_landmarks, output_size=(512, 512), type='landmarks')
process_images_in_directory(dir_affectnet, dir_affectnet_landmarksConeted, output_size=(512, 512), type='landmarksConeted')

#fer
process_images_in_directory(dir_fer, dir_fer_imagens_processed, output_size=(512, 512), type='imagens_processed')
process_images_in_directory(dir_fer, dir_fer_essenciais, output_size=(512, 512), type='essenciais')
process_images_in_directory(dir_fer, dir_fer_landmarks, output_size=(512, 512), type='landmarks')
process_images_in_directory(dir_fer, dir_fer_landmarksConeted, output_size=(512, 512), type='landmarksConeted')
