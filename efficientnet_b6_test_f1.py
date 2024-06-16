import numpy as np
import pandas as pd
import rasterio
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, custom_object_scope
from sklearn.metrics import confusion_matrix, f1_score
import json
import os
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

def Normalise(arr_band):
    return StandardScaler().fit_transform(arr_band.reshape(-1, 1)).reshape(arr_band.shape)

class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

model_path = "eurosat_efficientnetb6_best_ljeto(7).h5"
with custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model(model_path)
model.trainable = False

with open("EuroSATallBands/label_map.json", "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

def load_multispectral_image(img_path):
    with rasterio.open(img_path) as src:
        img = np.stack([src.read(i) for i in range(1, 14)], axis=-1)
    img = np.delete(img, [12,8,7,6,5,4], axis=-1)
    img = img.astype('float32')
    img /= 255.0
    return img

def prepare_data(csv_file, base_path):
    df = pd.read_csv(csv_file)
    X, y = [], []
    
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['Filename'])
        img = load_multispectral_image(img_path)
        X.append(img)
        y.append(class_names[row['ClassName']])
    
    X = np.array(X)
    y = to_categorical(y, num_classes=num_classes)
    
    return X, y

test_csv_file = "EuroSATallBands/test.csv"
base_path = "EuroSATallBands"
X_test, y_test = prepare_data(test_csv_file, base_path)

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

cnf_matrix = confusion_matrix(true_classes, predicted_classes)
f1_scores = f1_score(true_classes, predicted_classes, average=None)

print("Confusion Matrix:")
print(cnf_matrix)

print("\nF1 Scores per Class:")
for i, score in enumerate(f1_scores):
    class_name = list(class_names.keys())[i]
    print(f"Class {class_name}: {score:.3f}")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")
