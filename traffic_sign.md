# 信号認識

## Dataset

> !wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip

> !unzip GTSRB_Final_Training_Images.zip

## OpenCVのパッケージをいれる

> !apt-get update
> !apt-get -y install python-opencv

## Model

```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import csv
from matplotlib import pyplot as plt
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

batch_size = 128
num_classes = 42
epochs = 20

rows, cols = 25, 25

labels = []
features = []


for i in range(42):
    logs = []
    if i < 10:
        file_path = 'GTSRB/Final_Training/Images/0000' + str(i) 
        file_name = file_path + '/GT-0000' + str(i) + '.csv'
    elif i < 100:
        file_path = 'GTSRB/Final_Training/Images/000' + str(i) 
        file_name = file_path + '/GT-000' + str(i) + '.csv'

    with open(file_name,'rt') as file:
        reader = csv.reader(file, delimiter=";", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        for line in reader:
            logs.append(line)
        log_labels = logs.pop(0)

    for i in range(len(logs)):
        img_name = logs[i][0]
        img_path = file_path + '/' +  img_name
        img = plt.imread(img_path)
        #resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(cols,rows))
        resized = cv2.resize(img, (cols,rows))
        
        
        features.append(resized)
        labels.append(int(logs[i][7]))
        
print(len(features))
print(len(labels))

plt.imshow(features[0], cmap=plt.cm.gray_r,); plt.show()

features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

features = np.append(features,features[:,:,::-1],axis=0)
labels = np.append(labels,-labels,axis=0)

features, labels = shuffle(features, labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=0, test_size=0.1)

#reshape the data  to feed into the network
train_features = train_features.reshape(train_features.shape[0], rows, cols, 1)
test_features = test_features.reshape(test_features.shape[0], rows, cols, 1)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(rows, cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(train_features, train_labels, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          validation_data=(test_features, test_labels))
```



