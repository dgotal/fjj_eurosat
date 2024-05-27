import numpy as np
import pandas as pd
import rasterio
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score
import json
import os

def Normalise(arr_band):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(arr_band.reshape(-1, 1)).reshape(arr_band.shape)

model_path = "svi kanali/13bands_v5-5_SC.h5"
model = load_model(model_path)

with open("EuroSATallBands/label_map.json", "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

def prepare_data(csv_file, base_path, class_names):
    df = pd.read_csv(csv_file)
    X, y = [], []
    
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['Filename'])
        with rasterio.open(img_path) as src:
            bands_data = [src.read(band).astype(np.float32) for band in range(1, 14)]
            bands_data = [Normalise(band) for band in bands_data]
            bands_combined = np.dstack(bands_data)
            X.append(bands_combined)
        y.append(row['Label'])
    
    X = np.array(X)
    y = to_categorical(y, num_classes=num_classes)
    
    return X, y

test_csv_file = "EuroSATallBands/test.csv"
base_path = "EuroSATallBands"
X_test, y_test = prepare_data(test_csv_file, base_path, class_names)

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

cnf_matrix = confusion_matrix(true_classes, predicted_classes)
f1_scores = f1_score(true_classes, predicted_classes, average=None)

print("Confusion Matrix:")
print(cnf_matrix)

print("\nF1 Scores per Class:")
for i, score in enumerate(f1_scores):
    print(f"Class {list(class_names.keys())[i]}: {score:.3f}")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy of the model:", test_acc)
