import sys
import os
from torchvision import transforms
# Adicionar o caminho raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from src.data_processing.dataloader import DataLoaderSetup
from src.models.resnet_modelv2 import CustomResNet50
from src.training.trainer import Trainer

# Caminho do dataset
dataset_path = r'F:\Git\Teste\FER\affectnet\affectnet2'

# Definir dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")

# Transformações personalizadas
transformacoes_personalizadas = {
    'treino': transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validacao': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# Preparar DataLoaders
data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_personalizadas)
data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# Carregar o modelo
modelo = CustomResNet50(num_classes).get_model().to(device)

# Nome do modelo salvo e paciência para early stopping
nameModel = 'affectnet.pt'
patience = 5

# Treinar e validar
trainer = Trainer(modelo, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
epocas = 30
trainer.treinar_e_validar(epocas)
