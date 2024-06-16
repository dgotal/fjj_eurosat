import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import rasterio
import os
import seaborn as sns
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import custom_object_scope

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

model_path = 'eurosat_efficientnetb6_best_model(no6,8,13,5).h5'
with custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model(model_path)
model.trainable = False

def load_and_preprocess_image(path):
    with rasterio.open(path) as src:
        img = np.stack([src.read(i) for i in range(1, 14)], axis=-1)
    #img = np.delete(img, 5, axis=-1)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to compute the gradient importance
def compute_gradient_importance(model, input_tensor, class_index):
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        class_predictions = predictions[:, class_index]
    gradients = tape.gradient(class_predictions, input_tensor)
    gradients_reduced = tf.reduce_mean(gradients, axis=(1, 2))
    return gradients_reduced.numpy()

# Assuming you have 10 classes and a folder structure as mentioned
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
base_path = 'EuroSATallBands/'

channel_importances = np.zeros((len(classes), model.input.shape[-1]))

for class_index, class_name in enumerate(classes):
    for i in range(1, 51):  # Processing 50 images per class
        path = os.path.join(base_path, class_name, f'{class_name}_{i}.tif')
        image = load_and_preprocess_image(path)
        image = np.expand_dims(image, axis=0)  # Adding batch dimension
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        gradients = compute_gradient_importance(model, input_tensor, class_index)
        gradients = gradients.reshape(1, -1)
        channel_importances[class_index, :] += gradients[0, :]

normalized_importances = np.abs(channel_importances) / np.sum(np.abs(channel_importances), axis=1, keepdims=True)

df_normalized_importances = pd.DataFrame(normalized_importances, index=classes, columns=[f'Channel {i}' for i in range(1, model.input.shape[-1] + 1)])

plt.figure(figsize=(20, 10))
sns.heatmap(df_normalized_importances, annot=True, fmt=".2%", cmap="viridis")
plt.title('Relative Channel Importance by Class')
plt.xlabel('Channel')
plt.ylabel('Class')
plt.xticks(np.arange(model.input.shape[-1]) + 0.5, [f'Channel {i}' for i in range(1, model.input.shape[-1] + 1)])
plt.yticks(np.arange(len(classes)) + 0.5, classes)
plt.show()

average_importances = np.mean(normalized_importances, axis=0)

channel_to_exclude = np.argmin(average_importances)

print(f"Channel to exclude: Channel {channel_to_exclude + 1} with average importance: {average_importances[channel_to_exclude]:.2%}")

channels_sorted_by_importance = np.argsort(average_importances)

print("Channels sorted by average importance (least important first):")
for channel_index in channels_sorted_by_importance:
    print(f"Channel {channel_index + 1} - Average Importance: {average_importances[channel_index]:.2%}")
