import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

class Trainer:
    def __init__(
        self,
        model,
        data_loader_treino,
        data_loader_validacao,
        num_imagens_treino,
        num_imagens_validacao,
        device,
        num_classes=8,
        patience=5,
        nameModel='nome do arquivo.pt',
        otimizador=None,
        scheduler=None
    ):
        """
        Inicializa o Trainer com opções para usar ou não otimizador, scheduler e patience.

        Parâmetros:
        - model: O modelo a ser treinado.
        - data_loader_treino: DataLoader para o conjunto de treino.
        - data_loader_validacao: DataLoader para o conjunto de validação.
        - num_imagens_treino: Número total de imagens de treino.
        - num_imagens_validacao: Número total de imagens de validação.
        - device: Dispositivo onde o modelo será treinado (CPU ou GPU).
        - num_classes: Número de classes de saída.
        - patience: Número de épocas para o early stopping (opcional, pode ser None).
        - nameModel: Nome do arquivo para salvar o melhor modelo.
        - otimizador: O otimizador a ser usado (opcional, pode ser None).
        - scheduler: O scheduler de learning rate a ser usado (opcional, pode ser None).
        """
        self.model = model
        self.data_loader_treino = data_loader_treino
        self.data_loader_validacao = data_loader_validacao
        self.num_imagens_treino = num_imagens_treino
        self.num_imagens_validacao = num_imagens_validacao
        self.device = device
        self.funcao_erro = nn.CrossEntropyLoss()
        
        # Inicializa o otimizador apenas se for fornecido, senão cria um padrão
        if otimizador is None:
            self.otimizador = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        else:
            self.otimizador = otimizador
        
        # Inicializa o scheduler apenas se for fornecido
        self.scheduler = scheduler
        
        # Inicializa o patience
        self.patience = patience

        # Atualiza o caminho para salvar o modelo na pasta 'models/'
        self.nameModel = nameModel
        self.model_save_path = os.path.join('models', self.nameModel)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def treinar_e_validar(self, epocas):
        historico = []
        melhor_acuracia = 0.0
        early_stop_counter = 0

        for epoca in range(epocas):
            inicio_epoca = time.time()
            print(f"\n\nÉpoca: {epoca + 1}/{epocas}")
            erro_treino, acuracia_treino = self.executar_fase('treino')
            erro_validacao, acuracia_validacao, predicoes_validacao, labels_validacao = self.executar_fase('validacao', return_predictions=True)

            fim_epoca = time.time()
            print(f"Época {epoca + 1}/{epocas}, Treino: Erro: {erro_treino:.4f}, Acurácia: {acuracia_treino * 100:.2f}%, "
                  f"Validação: Erro: {erro_validacao:.4f}, Acurácia: {acuracia_validacao * 100:.2f}%, Tempo: {fim_epoca - inicio_epoca:.2f}s")

            historico.append([erro_treino, erro_validacao, acuracia_treino, acuracia_validacao])
            
            # Atualiza o scheduler se ele estiver definido e o scheduler não for None
            if self.scheduler is not None:
                self.scheduler.step(erro_validacao)

            # Early stopping
            if acuracia_validacao > melhor_acuracia:
                melhor_acuracia = acuracia_validacao
                print(f"Validation accuracy improved to {melhor_acuracia:.4f}. Saving the model.")
                try:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"Modelo salvo com sucesso em {self.model_save_path}")
                except Exception as e:
                    print(f"Erro ao salvar o modelo: {e}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if self.patience is not None and early_stop_counter >= self.patience:
                print("Parando o treinamento devido ao early stopping.")
                break

        # Calcular métricas finais
        self.calcular_metricas(predicoes_validacao, labels_validacao)
        return historico

    def treinar_e_validar(self, epocas):
        historico = []
        melhor_acuracia = 0.0
        early_stop_counter = 0

        for epoca in range(epocas):
            inicio_epoca = time.time()
            print(f"\n\nÉpoca: {epoca + 1}/{epocas}")
            erro_treino, acuracia_treino = self.executar_fase('treino')
            erro_validacao, acuracia_validacao, predicoes_validacao, labels_validacao = self.executar_fase('validacao', return_predictions=True)

            fim_epoca = time.time()
            print(f"Época {epoca + 1}/{epocas}, Treino: Erro: {erro_treino:.4f}, Acurácia: {acuracia_treino * 100:.2f}%, "
                  f"Validação: Erro: {erro_validacao:.4f}, Acurácia: {acuracia_validacao * 100:.2f}%, Tempo: {fim_epoca - inicio_epoca:.2f}s")

            historico.append([erro_treino, erro_validacao, acuracia_treino, acuracia_validacao])
            
            # Atualiza o scheduler se ele estiver definido
            if self.scheduler is not None:
                self.scheduler.step(erro_validacao)

            # Early stopping
            if acuracia_validacao > melhor_acuracia:
                melhor_acuracia = acuracia_validacao
                print(f"Validation accuracy improved to {melhor_acuracia:.4f}. Saving the model.")
                try:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"Modelo salvo com sucesso em {self.model_save_path}")
                except Exception as e:
                    print(f"Erro ao salvar o modelo: {e}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if self.patience is not None and early_stop_counter >= self.patience:
                print("Parando o treinamento devido ao early stopping.")
                break

        # Calcular métricas finais
        self.calcular_metricas(predicoes_validacao, labels_validacao)
        return historico

    def executar_fase(self, fase, return_predictions=False):
        if fase == 'treino':
            self.model.train()
            data_loader = self.data_loader_treino
            num_imagens = self.num_imagens_treino
        else:
            self.model.eval()
            data_loader = self.data_loader_validacao
            num_imagens = self.num_imagens_validacao

        erro_total = 0.0
        acuracia_total = 0.0
        todas_predicoes = []
        todas_labels = []

        with torch.set_grad_enabled(fase == 'treino'):
            print(f"\nExecutando a fase de {fase}...")
            for entradas, labels in data_loader:
                entradas, labels = entradas.to(self.device), labels.to(self.device)

                if fase == 'treino':
                    self.otimizador.zero_grad()

                saidas = self.model(entradas)
                erro = self.funcao_erro(saidas, labels)

                if fase == 'treino':
                    erro.backward()
                    self.otimizador.step()
                erro_total += erro.item() * entradas.size(0)
                _, predicoes = torch.max(saidas, 1)
                acuracia_total += (predicoes == labels).sum().item()

                if return_predictions:
                    todas_predicoes.extend(predicoes.cpu().numpy())
                    todas_labels.extend(labels.cpu().numpy())
                

        erro_medio = erro_total / num_imagens
        acuracia_media = acuracia_total / num_imagens

        if return_predictions:
            return erro_medio, acuracia_media, todas_predicoes, todas_labels
        else:
            return erro_medio, acuracia_media

    def calcular_metricas(self, predicoes, labels):
        acuracia = accuracy_score(labels, predicoes)
        precisao = precision_score(labels, predicoes, average='weighted', zero_division=0)
        recall = recall_score(labels, predicoes, average='weighted', zero_division=0)
        f1 = f1_score(labels, predicoes, average='weighted', zero_division=0)

        print("\nMétricas de Validação:")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\nRelatório de Classificação:")
        print(classification_report(labels, predicoes, zero_division=0))
