import pandas as pd
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from efficientnet.tfkeras import EfficientNetB6
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os

def create_efficientnetb6(n_classes, input_shape=(64, 64, 6)):
    base_model = EfficientNetB6(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

class EuroSATDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, csv_file, base_path, batch_size=32, dim=(64, 64), n_channels=0, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.base_path = base_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.df.iloc[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.df = pd.read_csv(self.csv_file)
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        for i, row in enumerate(list_IDs_temp):
            img_path = os.path.join(self.base_path, row['Filename'])
            img = self.load_multispectral_image(img_path)
            X[i,] = img
            y[i] = row['Label']
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def load_multispectral_image(self, img_path):
        with rasterio.open(img_path) as src:
            img = np.stack([src.read(i) for i in range(1, 14)], axis=-1)
        img = np.delete(img, 12, axis=-1)
        img = np.delete(img, 11, axis=-1)
        img = np.delete(img, 8, axis=-1)
        img = np.delete(img, 7, axis=-1)
        img = np.delete(img, 6, axis=-1)
        img = np.delete(img, 5, axis=-1)
        img = np.delete(img, 4, axis=-1)
        img = img.astype('float32')
        img /= 255.0
        return img

base_path = 'EuroSATallBands'
train_csv = 'EuroSATallBands/train.csv'
validate_csv = 'EuroSATallBands/validation.csv'
test_csv = 'EuroSATallBands/test.csv'
img_dim = (64, 64)
batch_size = 16
n_classes = 10
n_channels = 6

model = create_efficientnetb6(n_classes=n_classes, input_shape=(img_dim[0], img_dim[1], n_channels))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = EuroSATDataGenerator(csv_file=train_csv, base_path=base_path, batch_size=batch_size, dim=img_dim, n_channels=n_channels, n_classes=n_classes, shuffle=True)
validate_generator = EuroSATDataGenerator(csv_file=validate_csv, base_path=base_path, batch_size=batch_size, dim=img_dim, n_channels=n_channels, n_classes=n_classes, shuffle=False)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

model_checkpoint = ModelCheckpoint('eurosat_efficientnetb6_best_ljeto(7,12).h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')

history = model.fit(train_generator, validation_data=validate_generator, epochs=100, callbacks=[early_stopping, model_checkpoint])

model.save('eurosat_efficientnetb6_ljeto(7,12).h5')
