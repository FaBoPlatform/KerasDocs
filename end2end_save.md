# End2End Modelの作成

## モデルと学習済みデータの保存

```python
def model_save(model_json,model_h5):
    json_model = model.to_json()
    with open(model_json, "w") as f:
        f.write(json_model)
    model.save_weights(model_h5) 
```

> model_save("model.json", "model.h5")


## ここまでのソース

```python
import csv
from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

def load_data():
    folder_path = ''
    center_img = []
    direction_handle = []

    f = open(folder_path + 'driving_log.csv', 'r')

    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        img_name = row[0]
        img_path = folder_path  +  img_name
        img = plt.imread(img_path)
        center_img.append(img)
        direction_handle.append(float(row[3]))
    f.close()
    
    return (center_img, direction_handle)

def draw_img(center_img, direction_handle, pos=0):
    g_rows = 3
    g_cols = 3
    size = g_rows * g_cols
    fig, axs = plt.subplots(ncols=g_rows, nrows=size/g_rows)

    for h in range(g_rows):
        for i in range(g_cols):
            axs[h][i].imshow(center_img[pos + i * h], cmap=plt.cm.gray_r,) 
            axs[h][i].set_title(str(direction_handle[pos + h * i]))
    plt.show()

def process_img(center_img):
    new_center = []
    for img in center_img:
        crop_img = img[60:140, 0:320, :]
        resize_img = cv2.resize((cv2.cvtColor(crop_img, cv2.COLOR_RGB2XYZ))[:,:,2],(40,10))
        new_center.append(resize_img)
        
    return new_center    

def grow_data(center_img, direction_handle, delta):
    new_center = []
    new_direction = []
    for i in range(len(center_img)):
        for j in range(3):
            new_center.append(center_img[i])
            if j == 0:
                new_direction.append(direction_handle[i])
            elif j == 1:
                new_direction.append(direction_handle[i] + float(delta))
            elif j == 2:
                new_direction.append(direction_handle[i] - float(delta))
    return (new_center,  new_direction)

def end2end_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(rows,cols,1)))    

    model.add(Convolution2D(8, 3, 3, init='normal',border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),border_mode='valid'))

    model.add(Convolution2D(8, 3, 3,init='normal',border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),border_mode='valid'))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()

    return model

def model_save(model_json,model_h5):
    json_model = model.to_json()
    with open(model_json, "w") as f:
        f.write(json_model)
    model.save_weights(model_h5) 

if __name__ == '__main__':
    (center_img, direction_handle) = load_data()
    
    delta = 0.2
    (center_img, direction_handle) = grow_data(center_img, direction_handle, delta)

    center_data = process_img(center_img)
    
    draw_img(center_data, direction_handle, 50)

    center_data  = np.array(center_data).astype('float32')
    direction_handle = np.array(direction_handle).astype('float32')

    center_data, direction_handle = shuffle(center_data, direction_handle)

    train_center, test_center, train_direction, test_direction = train_test_split(center_data, direction_handle, random_state=0, test_size=0.1)

    rows = 10
    cols = 40
    train_center = train_center.reshape(train_center.shape[0], rows, cols, 1)
    test_center = test_center.reshape(test_center.shape[0], rows, cols, 1)

    batch = 128
    epoc = 10
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model = end2end()
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

    history = model.fit(train_center, train_direction,batch_size=batch, nb_epoch=epoc,verbose=1, validation_data=(test_center, test_direction))

    model_save("./model.json", "./model.h5")
```