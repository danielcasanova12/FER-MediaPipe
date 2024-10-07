import sys
import os
from torchvision import transforms
import torch

from src.models.resnet_modelv2 import CustomResNet50
from src.data_processing.dataloader import DataLoaderSetup
from src.models.resnet_V1 import ResnetV1
from src.training.trainer import Trainer


# Definir dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")

# Transformações personalizadas
transformacoes_modelo1 = {
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


transformacoes_modelo2 = {
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

# ###   Modelo 1 AFFECTENET

# # ## Modelo 1 - dados puros
# # dataset_path = r'./data/affectnet/processed/imagens_processed'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1affectnet.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - Essenciais
# # dataset_path = r'./data/affectnet/processed/essenciais'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1affectnetEssenciais.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - landmasks
# # dataset_path = r'./data/affectnet/processed/landmarks'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1affectnetlandmasks.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - landmarksConecteds
# # dataset_path = r'./data/affectnet/processed/landmarksConected'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1affectnetlandmarksConecteds.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)


# # ###   Modelo 1 FER

# # ## Modelo 1 - dados puros
# # dataset_path = r'./data/Fer-2013/processed/imagens_processed'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1Fer.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - Essenciais
# # dataset_path = r'./data/Fer-2013/processed/essenciais'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1FerEssenciais.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - landmasks
# # dataset_path = r'./data/Fer-2013/processed/landmarks'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1Ferlandmasks.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)

# # ## Modelo 1 - landmarksConecteds
# # dataset_path = r'./data/Fer-2013/processed/landmarksFullConected'
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo1)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # model = ResnetV1(num_classes=num_classes)
# # model = model.to(device)
# # nameModel = 'model1FerlandmarksConecteds.pt'
# # patience = 5

# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)




# # ## modelo 2 dados puros

# dataset_path = r'./data/affectnet/processed/imagens_processed'

# # Preparar DataLoaders
# data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # Carregar o modelo
# model = CustomResNet50(num_classes).get_model().to(device)

# # Nome do modelo salvo e paciência para early stopping
# nameModel = 'model2affectnet.pt'
# patience = 5

# # Treinar e validar
# trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# epocas = 30
# trainer.treinar_e_validar(epocas)


# ## modelo 2 essenciais

# dataset_path = r'./data/affectnet/processed/essenciais'

# # Preparar DataLoaders
# data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # Carregar o modelo
# model = CustomResNet50(num_classes).get_model().to(device)

# # Nome do modelo salvo e paciência para early stopping
# nameModel = 'model2affectnetessenciais.pt'
# patience = 5

# # Treinar e validar
# trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# epocas = 30
# trainer.treinar_e_validar(epocas)



# ## modelo 2 landmarks

# dataset_path = r'./data/affectnet/processed/landmarks'

# # Preparar DataLoaders
# data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # Carregar o modelo
# model = CustomResNet50(num_classes).get_model().to(device)

# # Nome do modelo salvo e paciência para early stopping
# nameModel = 'model2affectnetlandmarks.pt'
# patience = 5

# # Treinar e validar
# trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# epocas = 30
# trainer.treinar_e_validar(epocas)



# # modelo 2 fullconected

# dataset_path = r'./data/affectnet/processed/landmarksConected'

# # Preparar DataLoaders
# data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # Carregar o modelo
# model = CustomResNet50(num_classes).get_model().to(device)

# # Nome do modelo salvo e paciência para early stopping
# nameModel = 'model2affectnetfullconected.pt'
# patience = 5

# # Treinar e validar
# trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# epocas = 30
# trainer.treinar_e_validar(epocas)



# # ## modelo 2 dados puros

# # dataset_path = r'./data/Fer-2013/processed/imagens_processed'

# # # Preparar DataLoaders
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # # Carregar o modelo
# # model = CustomResNet50(num_classes).get_model().to(device)

# # # Nome do modelo salvo e paciência para early stopping
# # nameModel = 'model2Fer.pt'
# # patience = 5

# # # Treinar e validar
# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)


# # ## modelo 2 essenciais

# # dataset_path = r'./data/Fer-2013/processed/essenciais'

# # # Preparar DataLoaders
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # # Carregar o modelo
# # model = CustomResNet50(num_classes).get_model().to(device)

# # # Nome do modelo salvo e paciência para early stopping
# # nameModel = 'model2Feressenciais.pt'
# # patience = 5

# # # Treinar e validar
# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)



# # ## modelo 2 landmarks

# # dataset_path = r'./data/Fer-2013/processed/landmarks'

# # # Preparar DataLoaders
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # # Carregar o modelo
# # model = CustomResNet50(num_classes).get_model().to(device)

# # # Nome do modelo salvo e paciência para early stopping
# # nameModel = 'model2Ferlandmarks.pt'
# # patience = 5

# # # Treinar e validar
# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)



# # ## modelo 2 fullconected

# # dataset_path = r'./data/Fer-2013/processed/landmarksFullConected'

# # # Preparar DataLoaders
# # data_loader_setup = data_loader_setup = DataLoaderSetup(dataset_path,image_size=224,batch_size=32,transformacoes=transformacoes_modelo2)
# # data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes = data_loader_setup.get_data_loaders()

# # # Carregar o modelo
# # model = CustomResNet50(num_classes).get_model().to(device)

# # # Nome do modelo salvo e paciência para early stopping
# # nameModel = 'model2Ferfullconected.pt'
# # patience = 5

# # # Treinar e validar
# # trainer = Trainer(model, data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, device, num_classes, patience, nameModel)
# # epocas = 30
# # trainer.treinar_e_validar(epocas)
