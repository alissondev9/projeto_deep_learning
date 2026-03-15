# Deep Learning em Sensores Inteligentes 🚀

Este projeto implementa uma Rede Neural Convolucional (CNN) otimizada para o reconhecimento de imagens digitais, aplicada ao contexto de sensores inteligentes.

## 🛠️ Tecnologias
* **Python 3.x**
* **TensorFlow / Keras** (Processamento e modelagem)
* **NumPy** (Manipulação de arrays)
* **ImageDataGenerator** (Aumento de dados e pré-processamento)

## 🧠 Arquitetura Otimizada
Baseado no modelo proposto, foram aplicadas as seguintes melhorias:
1. **Dropout (0.5):** Redução de overfitting.
2. **Data Augmentation:** Rotação e zoom aleatórios para maior robustez.
3. **Camada Adicional:** Inclusão de uma camada Conv2D de 128 filtros para capturar detalhes complexos.

## 🚀 Como usar
1. Coloque suas imagens em `data/train/` (separadas por pastas de classes).
2. Execute `python src/train.py` para treinar o sensor.
3. Use `python src/predict.py` para classificar novas imagens.
