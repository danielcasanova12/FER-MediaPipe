import sys
import os
import torch
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from src.data_processing.dataloader import DataLoaderSetup
from src.training.trainer import Trainer

# Adicionar o caminho raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Caminho do dataset
dataset_path = r'F:\Git\Teste\FER\affectnet\affectnet2'

# Definir dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")

# Melhorar Transformações Personalizadas (Adicionando Cutout e RandomErasing)
transformacoes_personalizadas = {
    'treino': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ]),
    'validacao': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Preparar DataLoaders
data_loader_setup = DataLoaderSetup(
    dataset_path=dataset_path,
    image_size=224,
    batch_size=32,
    transformacoes=transformacoes_personalizadas
)
data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# Carregar o modelo EfficientNet-B4
model = models.efficientnet_b4(pretrained=True)

# Fine-tuning: Descongelar as últimas camadas convolucionais para treinamento
for param in model.parameters():
    param.requires_grad = False  # Congelar todas as camadas
for param in model.features[-3:].parameters():
    param.requires_grad = True  # Descongelar as últimas 3 camadas convolucionais

# Modificar a última camada fully connected para ajustar o número de classes
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, num_classes)
)

# Enviar o modelo para o dispositivo
model = model.to(device)

# Nome do modelo salvo e paciência para early stopping
nameModel = 'efficientnet_b4_affectnet.pt'
patience = 7

# Otimizador AdamW (melhor regularização) e Scheduler de LR baseado em métrica
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# Treinar e validar o modelo
trainer = Trainer(
    model=model,
    data_loader_treino=data_loader_treino,
    data_loader_validacao=data_loader_validacao,
    num_imagens_treino=num_imagens_treino,
    num_imagens_validacao=num_imagens_validacao,
    device=device,
    num_classes=num_classes,
    patience=patience,
    nameModel=nameModel,
    otimizador=optimizer,
    scheduler=scheduler
)

epocas = 50  # Podemos aumentar o número de épocas porque temos melhor controle com o scheduler e early stopping
trainer.treinar_e_validar(epocas)
