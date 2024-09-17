# import sys
# import os
# from torchvision import datasets, models, transforms
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Ensure the project root is in sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# from src.data_processing.dataloader import DataLoaderSetup
# from src.models.resnet_V1 import ResnetV1
# from src.training.trainer import Trainer

# # Number of classes
# numero_de_classes = 8
# model = ResnetV1(num_classes=numero_de_classes)

# # Define loss function and optimizer
# funcao_erro = nn.NLLLoss()
# otimizador = optim.Adam(model.parameters(), lr=0.0001)

# # Dataset paths
# dataset = r'F:\Git\Teste\FER\affectnet\affectnet2'

# nameModel = 'affectnet.pt'

# # Custom transformations
# transformacoes_personalizadas = {
#     'treino': transforms.Compose([
#         transforms.RandomRotation(15),
#         transforms.Resize(256),
#         transforms.ToTensor()
#     ]),
#     'validacao': transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor()
#     ])
# }

# # Initialize DataLoaderSetup with custom transformations
# data_loader_setup = DataLoaderSetup(
#     dataset_path=dataset,
#     image_size=224,
#     batch_size=32,
#     transformacoes=transformacoes_personalizadas
# )

# # Get data loaders
# data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # Verify data types
# data_iter = iter(data_loader_treino)
# images, labels = next(data_iter)
# print(f"Type of images: {type(images)}")   # Should be <class 'torch.Tensor'>
# print(f"Type of labels: {type(labels)}")   # Should be <class 'torch.Tensor'>

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Dispositivo utilizado: {device}")

# # Move the model to the device
# model = model.to(device)

# # Optionally, verify that the model is on the correct device
# for param in model.parameters():
#     print(f"Model parameter device: {param.device}")
#     break  # Check only the first parameter

# # Train and validate
# trainer = Trainer(
#     model=model,
#     data_loader_treino=data_loader_treino,
#     data_loader_validacao=data_loader_validacao,
#     num_imagens_treino=num_imagens_treino,
#     num_imagens_validacao=num_imagens_validacao,
#     device=device,
#     num_classes=num_classes,
#     name_model=nameModel,
#     otimizador=otimizador,
#     funcao_erro=funcao_erro
# )

# epocas = 30
# trainer.treinar_e_validar(epocas)
import sys
import os
from torchvision import transforms
# Adicionar o caminho raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import torch
from src.data_processing.dataloader import DataLoaderSetup
from src.models.resnet_V1 import ResnetV1
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
        transforms.Resize(256),
        transforms.ToTensor(),
    ]),
    'validacao': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
}


# Preparar DataLoaders
data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_personalizadas)
data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# Carregar o modelo
model = ResnetV1(num_classes=num_classes)
model = model.to(device)
# model = CustomResNet50(num_classes).get_model().to(device)

# Nome do modelo salvo e paciência para early stopping
nameModel = 'affectnet.pt'
patience = 5

# Treinar e validar
trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
epocas = 30
trainer.treinar_e_validar(epocas)
