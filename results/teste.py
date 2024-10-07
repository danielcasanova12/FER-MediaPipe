import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import os
import sys

# Adicionar o caminho do diretório 'src' no sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Pega o diretório do script atual
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # Sobe um nível no diretório
src_dir = os.path.join(parent_dir, 'src')  # Adiciona a pasta 'src'
sys.path.append(src_dir)

# Importações dos módulos personalizados
from models.resnet_modelv2 import CustomResNet50
from data_processing.dataloader import DataLoaderSetup
from models.resnet_V1 import ResnetV1


print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA não está disponível. Usando CPU.")


# Funções de Transformações por Modelo
def get_transforms(model_version):
    if model_version == 1:
        return {
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
    elif model_version == 2:
        return {
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

# Função para testar o modelo
def test_model(model, test_loader, device):
    model.eval()  # Modo de avaliação
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calcular as métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, report


# Função para carregar e configurar o modelo correto
def load_model_from_checkpoint(checkpoint, model_class, model_version):
    if model_class == CustomResNet50:
        # Inicializar o modelo CustomResNet50
        model = CustomResNet50(num_classes=None).get_model()  # Inicializa sem classes na última camada
        in_features = model.fc.in_features  # Obtém a entrada da fully connected layer
        
        # Verificar a existência da camada 'fc.weight' no checkpoint
        if 'fc.3.weight' in checkpoint:
            num_classes_checkpoint = checkpoint['fc.3.weight'].shape[0]  # Para CustomResNet50
            model.fc = nn.Linear(in_features, num_classes_checkpoint)  # Ajustar a última camada
        else:
            raise KeyError("Checkpoint não contém 'fc.3.weight'. Certifique-se de que o checkpoint contém os pesos da fully connected layer.")
    
    elif model_class == ResnetV1:
        # Para ResnetV1
        if 'resnet.fc.3.weight' in checkpoint:
            num_classes_checkpoint = checkpoint['resnet.fc.3.weight'].shape[0]  # Para ResnetV1
            model = model_class(num_classes=num_classes_checkpoint).to(device)
        else:
            raise KeyError("Checkpoint não contém 'resnet.fc.3.weight'. Certifique-se de que o checkpoint contém os pesos da fully connected layer.")

    return model


# Função principal para carregar e testar o modelo
def load_and_test_model(dataset_path, model_class, model_path, model_version, image_size=224, batch_size=32, results_file="results.txt"):
    try:
        # Configurar as transformações para os dados de teste
        transformacoes = get_transforms(model_version)
        
        # Carregar os dados de teste
        data_loader_setup = DataLoaderSetup(dataset_path=dataset_path, image_size=image_size, batch_size=batch_size, transformacoes=transformacoes)
        _, test_loader, _, _, _ = data_loader_setup.get_data_loaders()
        
        # Carregar o estado salvo do modelo
        checkpoint = torch.load(model_path, map_location=device)
        
        # Verificação do conteúdo do checkpoint
        print(f"Conteúdo do checkpoint: {checkpoint.keys()}")

        # Inicializar o modelo com base no tipo de modelo e carregar os pesos
        model = load_model_from_checkpoint(checkpoint, model_class, model_version)

        # Carregar o estado do modelo
        model.load_state_dict(checkpoint)
        
        # Testar o modelo
        accuracy, precision, recall, f1, report = test_model(model, test_loader, device)
        
        # Exibir e salvar os resultados
        print(f"\nResultados para o modelo {model_path}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Classification Report:\n{report}")
        
        with open(results_file, "a") as f:
            f.write(f"\nResultados para o modelo {model_path}:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write("="*80 + "\n")
    
    except KeyError as ke:
        # Se ocorrer um erro de chave, salvar o erro no arquivo de resultados
        print(f"Erro de chave ao testar o modelo {model_path}: {ke}")
        with open(results_file, "a") as f:
            f.write(f"\nErro de chave ao testar o modelo {model_path}:\n")
            f.write(f"{str(ke)}\n")
            f.write("="*80 + "\n")
    
    except Exception as e:
        # Se ocorrer qualquer outro erro, salvar o erro no arquivo de resultados
        print(f"Erro ao testar o modelo {model_path}: {e}")
        with open(results_file, "a") as f:
            f.write(f"\nErro ao testar o modelo {model_path}:\n")
            f.write(f"{str(e)}\n")
            f.write("="*80 + "\n")


# Configurações dos modelos e datasets
model_test_configs = [
    {'dataset_path': './data/Fer-2013/processed/imagens_processed', 'model_class': CustomResNet50, 'model_path': 'f:/Git/Teste/FER-MediaPipe/models/model2Fer.pt', 'model_version': 2},
    {'dataset_path': './data/Fer-2013/processed/essenciais', 'model_class': CustomResNet50, 'model_path': 'f:/Git/Teste/FER-MediaPipe/models/model2FerEssenciais.pt', 'model_version': 2},
    {'dataset_path': './data/Fer-2013/processed/landmarks', 'model_class': CustomResNet50, 'model_path': 'f:/Git/Teste/FER-MediaPipe/models/model2FerLandmarks.pt', 'model_version': 2},
    {'dataset_path': './data/Fer-2013/processed/landmarksFullConected', 'model_class': CustomResNet50, 'model_path': 'f:/Git/Teste/FER-MediaPipe/models/model2FerFullConected.pt', 'model_version': 2},
]

# Nome do arquivo de resultados
results_file = "model_results.txt"

# Testar todos os modelos e salvar os resultados
for config in model_test_configs:
    print(f"\nTestando o modelo: {config['model_path']}")
    load_and_test_model(config['dataset_path'], config['model_class'], config['model_path'], config['model_version'], results_file=results_file)
