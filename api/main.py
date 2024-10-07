from fastapi import FastAPI, File, UploadFile
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models/')))

from resnet_modelv2 import CustomResNet50



import torch
import torchvision.transforms as transforms
from io import BytesIO

# Inicializar FastAPI
app = FastAPI()

# Definir as classes de emoção
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Definir o dispositivo para execução (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo treinado (ajustar conforme sua arquitetura)
modelo_treinado = CustomResNet50(8).get_model()  # Obtenha seu modelo aqui
modelo_treinado.load_state_dict(torch.load('../models/model2affectnet.pt', map_location=device))  # Carregar pesos
modelo_treinado.eval()  # Colocar o modelo em modo de avaliação
modelo_treinado.to(device)

# Definir transformações da imagem
image_size = 224  # Ajustar conforme necessário
transformacao = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Função para prever a emoção
def prever_emocao(image: Image.Image):
    # Aplicar transformações na imagem
    imagem_transformada = transformacao(image).unsqueeze(0).to(device)
    
    # Fazer predição
    with torch.no_grad():
        saida = modelo_treinado(imagem_transformada)
        _, pred = torch.max(saida, 1)
        emocao_predita = classes[pred.item()]
    
    return emocao_predita

# Endpoint para receber imagem e retornar a predição da emoção
@app.post("/predizer-emocao/")
async def predizer_emocao(file: UploadFile = File(...)):
    # Ler a imagem enviada
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")  # Converter para RGB

    # Prever a emoção
    emocao = prever_emocao(image)

    # Retornar a emoção predita
    return {"emocao_predita": emocao}

