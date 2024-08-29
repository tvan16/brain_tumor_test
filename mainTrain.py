
import cv2  
import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
INPUT_SIZE = 256
image_directory = 'datasets/'

no_tumor_images=os.listdir(image_directory+ 'crop_no/')
yes_tumor_images=os.listdir(image_directory+ 'crop_yes/')
dataset=[]
label=[]
# print(no_tumor_images)
# print(yes_tumor_images)

for i , image_name in enumerate(no_tumor_images) :
    if(image_name.split('.')[1] == 'jpg') :
        image=cv2.imread(image_directory+ 'crop_no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images) :
    if(image_name.split('.')[1] == 'jpg') :
        image=cv2.imread(image_directory+ 'crop_yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 ) #chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỷ lệ 80% và 20%.

# print(x_train, x_test, y_train, y_test)
#Reshape = (n, image_width, image_height, n_channel)

#print
# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

print(tf.__version__)

from sklearn.preprocessing import normalize
import numpy as np
x_train = np.array(x_train, dtype=float)
x_test= np.array(x_test, dtype=float)
print(x_train.shape)

x_train_normalized=x_train/255 # chia khoảng từ 0-1
x_test_normalized=x_test/255 # chia khoảng từ 0-1

y_train=to_categorical(y_train, num_classes=2) # đánh nhãn label theo dạng one-hot vector
y_test=to_categorical(y_test, num_classes=2) 


print(x_train)
print(x_train_normalized)

#Model Building
#64,64,3
model=Sequential() # tuần tự nghĩa là các lớp được thêm vào theo thứ tự câu lệnh
model.add(Conv2D(32, (3, 3),input_shape=(INPUT_SIZE, INPUT_SIZE, 3))) # gồm 3 lớp convolution xen kẽ với lớp pooling
model.add(Activation('relu')) #công thức tính outpout 1 layer là A^l = activation(W^l x A^l-1 + B^l), relu :  max(0,x) 
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu')) # ví sao dùng activation: tạo sự phi tuyến tính + giữ output trong khoảng nhất định (0,1)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten()) # duỗi thẳng 
model.add(Dense(64)) # đưa qua 1 lớp 64 neuron
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(Dense(2)) # đưa qua lớp 2 neuron là : yes no
model.add(Activation('sigmoid')) # 1/(1 + e^-x)

#BinaryCrossEntryopy = 1, sigmoid
#CateCrossEntryopy = 2, softmax thường ở cuối
###leaningrate
# Đặt learning rate mong muốn
# quyết định kích thước của các bước mà mô hình sẽ di chuyển trên mặt phẳng gradient để tối ưu hóa hàm loss
optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-3) 
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']  
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
import log
file_logging_callback  = log.FileLoggingCallback()

model.fit(
    x_train_normalized, 
    y_train, 
    batch_size=16,
    verbose=1,
    epochs=10, 
    validation_data=(x_test_normalized, y_test),
    shuffle=False,
    callbacks=[file_logging_callback, reduce_lr, early_stopping]
)

model.save('Braintumor10EpochsCategorical.h5')








