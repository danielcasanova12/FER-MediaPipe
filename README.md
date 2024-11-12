# FER-MediaPipe

[Link para a previsão em tempo real](https://fer-01.vercel.app/) 
Este repositório apresenta o projeto **Reconhecimento de Emoções Faciais (FER)** usando MediaPipe para a detecção de pontos faciais e modelos personalizados para a classificação de emoções, com uma precisão impressionante de **72%** em imagens puras!
## Tabela de Conteúdos

- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso](#uso)
- [Modelos](#modelos)
  - [Modelo 1: ResNetV1](#modelo-1-resnetv1)
  - [Modelo 2: CustomResNet50](#modelo-2-customresnet50)
- [Datasets](#datasets)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Visão Geral do Projeto

O projeto **FER-MediaPipe** tem como objetivo classificar emoções faciais com base em pontos detectados através do **MediaPipe**. Este modelo detecta pontos faciais chave e os envia para um classificador treinado para prever um dos vários estados emocionais.

O objetivo principal deste projeto foi comparar o desempenho de diferentes tipos de **landmarks** oferecidos pelo **MediaPipe** com classificadores de deep learning para prever emoções humanas.

Para isso, usamos dois datasets populares: o [AffectNet](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) e o [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).

### Comparação de Tipos de Landmarks e a Imagem Original

**Imagem original:**

![Imagem pura](Imagens/imagens_puras/image0000697.jpg)

O MediaPipe oferece diferentes tipos de landmarks, como pontos básicos, pontos detalhados e mesh completo. Abaixo está uma comparação visual desses pontos:

- **Landmarks Básicos:** Conjunto mínimo de pontos que inclui olhos, boca e contorno do rosto.
  
  ![landmarks](Imagens/landmarks_basicos/img12.jpg)
- **Landmarks Mesh:** O mesh completo oferece uma representação densa da face, com pontos espalhados por toda a superfície.
  
  ![landmarks completos](Imagens/landmarks_completos/img12.jpg)

- **Landmarks Detalhados:** Conjunto ampliado que inclui sobrancelhas, nariz e boca de forma mais precisa.
  
  ![landmarks detalhados](Imagens/landmarks_detalhados/img12.jpg)



As emoções detectadas incluem:
- Raiva
- Desprezo
- Nojo
- Medo
- Felicidade
- Neutro
- Tristeza
- Surpresa

## Funcionalidades

- Extração de pontos faciais usando [MediaPipe](https://google.github.io/mediapipe/).
- Classificação de emoções usando modelos de deep learning personalizados.
- Integração com datasets populares como FER-2013 e AffectNet.
- Detecção de emoções faciais em tempo real via webcam ou imagens estáticas.
- Comparação entre diferentes tipos de landmarks faciais.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/danielcasanova12/FER-MediaPipe.git
    cd FER-MediaPipe
    ```

2. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

3. Instale as dependências adicionais para o MediaPipe e machine learning:
    ```bash
    pip install mediapipe opencv-python tensorflow
    ```

4. Se encontrar problemas com o `dlib`, certifique-se de que o `CMake` está instalado e configurado no PATH do sistema.

## Uso

### Execução da Detecção de Emoções com Webcam

Para executar a detecção de emoções em tempo real usando sua webcam, use o seguinte comando:

```bash
python webcam_emotion_detector.py
```

### Execução da Detecção em Imagens Estáticas

Para processar uma imagem específica e prever a emoção, utilize o comando:

```bash
python image_emotion_detector.py --image_path "caminho/para/imagem.jpg"
```

## Modelos

### Modelo 1: ResNetV1

O **Modelo 1** é uma versão personalizada da ResNet-50, com algumas camadas descongeladas para ajuste fino.

**Características:**
- Modelo Base: ResNet-50 pré-treinada no ImageNet.
- Congelamento de Camadas: Todas as camadas, exceto as últimas 4, foram congeladas.
- Modificação da Camada Final:
  - `Linear(num_features, 512)`
  - `ReLU(inplace=True)`
  - `Dropout(0.3)`
  - `Linear(512, num_classes)`
  - `LogSoftmax(dim=1)`

### Modelo 2: CustomResNet50

O **Modelo 2** também utiliza a arquitetura ResNet-50, mas com todas as camadas treináveis.

**Características:**
- Modelo Base: ResNet-50 pré-treinada no ImageNet.
- Todas as Camadas Treináveis: Nenhuma camada foi congelada.
- Modificação da Camada Final:
  - `Linear(num_features, 512)`
  - `ReLU()`
  - `Dropout(0.5)`
  - `Linear(512, num_classes)`

## Datasets

### FER-2013

O dataset **FER-2013** foi pré-processado para remover a classe **"Nojo"**, simplificando a tarefa de classificação. As classes restantes incluem:
- Raiva
- Medo
- Felicidade
- Tristeza
- Surpresa
- Neutro

### AffectNet

O dataset **AffectNet** foi utilizado sem modificações. Contém 8 classes de emoções, incluindo a classe "Desprezo".

## Estrutura do Projeto

```bash
├── data                     # Pasta com os datasets
├── models                   # Modelos treinados
├── src                      # Código fonte
│   ├── model_1.py           # Definição do Modelo 1
│   ├── model_2.py           # Definição do Modelo 2
│   ├── train.py             # Script de treinamento
│   └── transforms.py        # Transformações dos dados
├── requirements.txt         # Dependências do projeto
├── README.md                # Este arquivo
```
## Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests se tiver sugestões de melhorias ou encontrar problema
