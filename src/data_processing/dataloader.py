from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class DataLoaderSetup:
    def __init__(self, dataset_path, image_size, batch_size, transformacoes=None):
        """
        Inicializa a classe DataLoaderSetup.

        Parâmetros:
        - dataset_path: Caminho para o dataset.
        - image_size: Tamanho da imagem para redimensionamento.
        - batch_size: Tamanho do batch.
        - transformacoes: Dicionário opcional com as transformações para 'treino' e 'validacao'. Se None, serão usadas transformações padrão.
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Se não forem fornecidas transformações, usa as transformações padrão
        self.transformacoes = transformacoes or self.get_default_transforms()

    def get_default_transforms(self):
        """
        Define transformações padrão para o conjunto de treino e validação.
        """
        transformacoes_de_imagens = {
            'treino': transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size)
            ]),
            'validacao': transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size)
            ])
        }
        return transformacoes_de_imagens

    def get_data_loaders(self):
        """
        Cria DataLoaders para os conjuntos de treino e validação, aplicando as transformações fornecidas ou padrão.
        """
        # Usa as transformações fornecidas ou as padrão definidas no init
        train_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'treino'), transform=self.transformacoes['treino'])
        val_dataset = datasets.ImageFolder(os.path.join(self.dataset_path, 'validacao'), transform=self.transformacoes['validacao'])

        data_loader_treino = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        data_loader_validacao = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        num_imagens_treino = len(train_dataset)
        num_imagens_validacao = len(val_dataset)
        num_classes = len(train_dataset.classes)

        return data_loader_treino, data_loader_validacao, num_imagens_treino, num_imagens_validacao, num_classes
