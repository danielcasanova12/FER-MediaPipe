# FER-MediaPipe

Este repositório contém um projeto de Reconhecimento de Emoções Faciais (FER) usando MediaPipe para a detecção de pontos faciais e modelos personalizados para a classificação de emoções.
O objetivo principal deste projeto foi comparar o desempenho de diferentes landmarsk com o <a href=“https://ai.google.dev/edge/mediapipe/solutions/guide?hl=pt-br“>MediaPipe</a>
## Tabela de Conteúdos

- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso](#uso)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Visão Geral do Projeto

O projeto **FER-MediaPipe** tem como objetivo classificar emoções faciais com base em pontos detectados através do MediaPipe. Este modelo detecta pontos faciais chave e os envia para um classificador treinado para prever um dos vários estados emocionais.

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
- Classificação de emoções usando um modelo de deep learning personalizado.
- Integração com datasets populares como FER2013 e AffectNet.
- Detecção de emoções faciais em tempo real via webcam ou imagens estáticas.

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
