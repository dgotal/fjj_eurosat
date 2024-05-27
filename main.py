import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, itertools, rasterio, math, warnings
from tensorflow.keras import backend as K
from tqdm import tqdm
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot, to_categorical
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, Concatenate, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

def show_final_history(history):
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    
    ax[0].plot(history.history['loss'], 'r-', label='Training Loss')
    ax[0].plot(history.history['val_loss'], 'g-', label='Validation Loss')
    ax[1].plot(history.history['categorical_accuracy'], 'r-', label='Training Accuracy')
    ax[1].plot(history.history['val_categorical_accuracy'], 'g-', label='Validation Accuracy')
    
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')
    
    plt.show()
    pass

def plot_learning_rate(loss_history):
    plt.style.use("ggplot")
    plt.plot(np.arange(len(loss_history.lr)), loss_history.lr)
    plt.title("Learning Rate over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.0
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j,i, format(cm[i,j], fmt),
                horizontalalignment = 'center',
                color = "white" if cm[i,j] > thresh else "black")
        pass
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.grid(False)
    pass

with open("G:/Obrada slike/osirv_projekt/EuroSATallBands/label_map.json","r") as f:
    class_names_encoded = json.load(f)
    pass

class_names = list(class_names_encoded.keys())
num_classes = len(class_names)
class_names_encoded

bands = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'8a':9,'9':10,'10':11,'11':12,'12':13}

def Normalise(arr_band):
    
    return StandardScaler().fit_transform(arr_band)

basePath = "G:/Obrada slike/osirv_projekt/EuroSATallBands"

def data_generator_svi_kanali(csv_file, num_classes, batch_size = 10, target_size = 64):
    df = pd.read_csv(csv_file)
    num_samples = df.shape[0]

    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples_idx = df.index[offset:offset+batch_size]
            X, y = [], []
            for i in batch_samples_idx:
                img_name = df.loc[i,'Filename']
                label = df.loc[i,'Label']
                src = rasterio.open(os.path.join(basePath, img_name))
                bands_data = [src.read(band).astype(np.float32) for band in range(1, 14)]
                bands_data = [Normalise(band) for band in bands_data]
                bands_combined = np.dstack(bands_data)
                X.append(bands_combined)
                y.append(label)
            X = np.array(X)
            y = np.array(y)
            y = to_categorical(y, num_classes=num_classes)
            yield X, y
    pass

def data_generator(csv_file, num_classes, batch_size=10, target_size=64): 
    df = pd.read_csv(csv_file)
    num_samples = df.shape[0]
    
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples_idx = df.index[offset:offset+batch_size]

            X, y = [], []

            for i in batch_samples_idx:
                img_name = df.loc[i, 'Filename']
                label = df.loc[i, 'Label']

                src = rasterio.open(os.path.join(basePath, img_name))

                bands_data = [src.read(band).astype(np.float32) for band in range(1, 14) if band != 2 and band != 4 and band != 6 and band != 8 and band != 11 and band != 12 and band != 13]
                bands_data = [Normalise(band) for band in bands_data]
                bands_combined = np.dstack(bands_data)

                X.append(bands_combined)
                y.append(label)

            X = np.array(X)
            y = np.array(y)
            y = to_categorical(y, num_classes=num_classes)
            
            yield X, y

train_generator = data_generator(csv_file = "G:/Obrada slike/osirv_projekt/EuroSATallBands/train.csv", num_classes = 10, batch_size = 10)
val_generator = data_generator(csv_file = "G:/Obrada slike/osirv_projekt/EuroSATallBands/validation.csv", num_classes = 10, batch_size = 10)

train_df = pd.read_csv("G:/Obrada slike/osirv_projekt/EuroSATallBands/train.csv")
train_labels = train_df.loc[:,'Label']
train_labels = np.array(train_labels)

num_train_samples = train_labels.shape[0]

val_df = pd.read_csv("G:/Obrada slike/osirv_projekt/EuroSATallBands/validation.csv")
val_labels = val_df.loc[:,'Label']
val_labels = np.array(val_labels)

num_val_samples = val_labels.shape[0]

num_train_samples, num_val_samples

def spectral_block(X,filters,stage,s=1):
    
    squeeze_base_name = 'squeeze_' + str(stage) + '_branch'
    bn_base_name = 'bn_' + str(stage) + "_branch"
    
    F1,F2,F3 = filters
    
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='same', name=squeeze_base_name+'a')(X)
    
    X_11 = Conv2D(filters=F2, kernel_size=(1,1), strides=(s,s), padding='same', name=squeeze_base_name+'b')(X)
    X_33 = Conv2D(filters=F3, kernel_size=(3,3), strides=(s,s), padding='same', name=squeeze_base_name+'c')(X)
    
    X = Concatenate(name="concatenate_"+str(stage))([X_11, X_33])
    X = BatchNormalization(name=bn_base_name)(X)
    
    X = Activation("relu", name="spectral"+str(stage))(X)
    
    return X


def SpectrumNet(input_shape, classes):
    
    X_input = Input(input_shape, name="input")
    
    X = Conv2D(96, (1,1), strides=(2,2), name='conv1', padding="same")(X_input)
    
    X = spectral_block(X, [16,96,32], 2)
    X = spectral_block(X, [16,96,32], 3)
    X = spectral_block(X, [32,192,64], 4)
    
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same", name="maxpool4")(X)
    
    X = spectral_block(X, [32,192,64], 5)
    X = spectral_block(X, [48, 288, 96], 6)
    X = spectral_block(X, [48, 288, 96], 7)
    X = spectral_block(X, [64, 384, 128], 8)
    
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same", name="maxpool8")(X)
    
    X = spectral_block(X, [64,384,128], 9)
    
    X = Conv2D(10, kernel_size=(1,1), strides=(1,1), name="conv10", padding='same')(X)
    X = BatchNormalization(name="conv10_batchnormalisation")(X)
    X = Activation("relu", name="conv10_activation")(X)
    
    X = AveragePooling2D(pool_size=(8,8), strides=(1,1), name="avgpool10")(X)
    
    X = Flatten(name="flatten10")(X)

    X = Activation("softmax", name="output")(X)
    
    model = Model(inputs=X_input, outputs=X, name="SpectrumNet")
    
    return model
    pass

model = SpectrumNet(input_shape = (64,64,6), classes=num_classes)

model.summary()

SVG(model_to_dot(model).create(prog="dot", format='svg'))

checkpoint = ModelCheckpoint("w_los_model(10,1,9,5,7,3).h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
logs = TensorBoard("13bands-logs-SC", histogram_freq=1)

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.10
   epochs_drop = 30.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.lr = []
    
    def on_epoch_end(self, epoch, logs={}):
        current_lr = K.get_value(self.model.optimizer.lr)
        self.lr.append(current_lr)

loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)

train_labels_encoded = to_categorical(train_labels,num_classes=10)

classTotals = train_labels_encoded.sum(axis=0)
classWeight = {}

for i in range(len(classTotals)):
    classWeight[i] = classTotals.max()/classTotals[i]
    pass

classWeight

opt = SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model = SpectrumNet(input_shape = (64,64,6), classes=num_classes)
model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

epochs = 100
batchSize = 100
  
history = model.fit(
    train_generator,
    steps_per_epoch = num_train_samples // batchSize,
    epochs = epochs,
    verbose = 1,
    validation_data = val_generator,
    validation_steps = num_val_samples // batchSize,
    callbacks = [checkpoint, LearningRateScheduler(step_decay), loss_history]
)

def obtain_tif_images(csv_file):
    df = pd.read_csv(csv_file)
    num_samples = df.shape[0]
    
    X, y = [], []
    
    for i in tqdm(range(num_samples)):
        img_name = df.loc[i, 'Filename']
        label = df.loc[i, 'Label']

        src = rasterio.open(os.path.join(basePath, img_name))
        
        bands_data = [src.read(band).astype(np.float32) for band in range(1, 14) if band != 2 and band != 4 and band != 6 and band != 8 and band != 11 and band != 12 and band != 13]
        bands_data = [Normalise(band) for band in bands_data]
        bands_combined = np.dstack(bands_data)

        X.append(bands_combined)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y


test_tifs, test_labels = obtain_tif_images(csv_file="G:/Obrada slike/osirv_projekt/EuroSATallBands/test.csv")

test_labels_encoded = to_categorical(test_labels, num_classes = len(class_names))

test_tifs.shape, test_labels.shape, test_labels_encoded.shape

test_pred = model.predict(test_tifs)
test_pred = np.argmax(test_pred, axis=1)
test_pred.shape

cnf_mat = confusion_matrix(test_labels, test_pred)

plot_confusion_matrix(cnf_mat, classes=class_names, title="5 Bands Confusion Matrix - V5-5 - SC")
plt.grid(False);

for f1,class_name in zip(f1_score(test_labels, test_pred, average=None), class_names):
    print("Class name: {}, F1 score: {:.3f}".format(class_name, f1))
    pass

model.save("los_model(10,1,9,5,7,3).h5")

model_test = load_model("G:/Obrada slike/osirv_projekt/los_model(10,1,9,5,7,3).h5")

model_test.summary()

model_test.load_weights("G:/Obrada slike/osirv_projekt/w_los_model(10,1,9,5,7,3).h5")

test_pred_2 = model_test.predict(test_tifs)
test_pred_2 = np.argmax(test_pred_2, axis=1)
test_pred_2.shape

for f1,class_name in zip(f1_score(test_labels, test_pred_2, average=None), class_names):
    print("Class name: {}, F1 score: {:.3f}".format(class_name, f1))
    pass

cnf_mat = confusion_matrix(test_labels, test_pred_2)

plot_confusion_matrix(cnf_mat, classes=class_names, title="Testing Model V5-5 CNF-MAT")
plt.grid(True)

val_tifs, val_labels = obtain_tif_images(csv_file="G:/Obrada slike/osirv_projekt/EuroSATallBands/validation.csv")

val_labels_encoded = to_categorical(val_labels, num_classes = len(class_names))

val_tifs.shape, val_labels.shape, val_labels_encoded.shape

val_pred = model_test.predict(val_tifs)
val_pred = np.argmax(val_pred, axis=1)
val_pred.shape

for f1,class_name in zip(f1_score(val_labels, val_pred, average=None), class_names):
    print("Class name: {}, F1 score: {:.3f}".format(class_name, f1))
    pass

cnf_mat = confusion_matrix(val_labels, val_pred)

plot_confusion_matrix(cnf_mat, classes=class_names,title="Validation Model V5-5 CNF-MAT")
plt.grid(True)