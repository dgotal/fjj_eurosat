import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import rasterio

model_path = 'eurosat_efficientnetb6_best_model.h5'
model = load_model(model_path)
model.trainable = False

num_classes = 10

def load_and_preprocess_image(path, exclude_channels=[]):
    with rasterio.open(path) as src:
        image = src.read()
        image = np.delete(image, [ch - 1 for ch in exclude_channels], axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = image.astype('float32')
        image = (image - image.min()) / (image.max() - image.min())
    return image

def compute_gradient_importance(model, input_tensor, class_index):
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor, training=False)
        class_predictions = predictions[:, class_index]
    gradients = tape.gradient(class_predictions, input_tensor)
    gradients_reduced = tf.reduce_mean(gradients, axis=(1, 2))
    return gradients_reduced

image_paths = []
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'] 

for class_name in classes:
    for i in range(1, 500):
        path = f'EuroSATallBands/{class_name}/{class_name}_{i}.tif'
        image_paths.append(path)

images = np.array([load_and_preprocess_image(p) for p in image_paths])
images = np.expand_dims(images, axis=0)  

channel_importances = np.zeros((num_classes, model.input.shape[-1]))

for class_index in range(num_classes):
    print(f"Computing for class index: {class_index}")
    image = load_and_preprocess_image(image_paths[class_index])
    image = np.expand_dims(image, axis=0)
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    gradients = compute_gradient_importance(model, input_tensor, class_index)
    channel_importances[class_index, :] = gradients[0] if len(gradients.shape) > 1 else gradients

df_importances = pd.DataFrame(channel_importances, columns=[f'Channel {i+1}' for i in range(channel_importances.shape[1])],
                              index=[f'Class {i+1}' for i in range(num_classes)])

print(df_importances)

import seaborn as sns

df_importances_abs = df_importances.abs()
df_normalized = df_importances_abs.div(df_importances_abs.sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
sns.heatmap(df_normalized, annot=True, fmt=".2%", cmap="YlGnBu")
plt.title('Relativna važnost kanala po klasama')
plt.xlabel('Kanal')
plt.ylabel('Klasa')
plt.show()

average_importances = df_normalized.mean(axis=0)

lowest_importance_channel = average_importances.idxmin()

print(f"Kanal za uklanjanje: {lowest_importance_channel}, s prosječnom važnošću: {average_importances[lowest_importance_channel]:.2%}")