import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

def load_and_predict(image_path, model_path='../models/modelo.keras'):
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 #

    prediction = model.predict(img_array)
    
    class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    print(f"Resultado da Análise: Classe {class_idx}")
    print(f"Confiança do Sensor: {confidence:.2f}%")

if __name__ == "__main__":
    test_image = "data/imagem_teste.jpg"
    load_and_predict(test_image)