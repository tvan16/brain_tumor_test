import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('Braintumor10EpochsCategorical.h5')

image=cv2.imread(r'C:/Users/Vu The Van/OneDrive/Documents/VSCode/BRT350 - Copy/brain_tumor/test.jpg')

img=Image.fromarray(image)

img=img.resize((256, 256))

img=np.array(img)

print(img)

input_img=np.expand_dims(img, axis=0)

predictions=model.predict(input_img)
result = np.argmax(predictions, axis=1)
print(result)


# TẠI SAO THƯỜNG DÙNG HÀM TANH RELU Ở GIỮA CÒN DÙNG SOFTMAX Ở CUỐI
# CHẠY TRÊN F1 SCORE
# CẮT DATA DẠNG HÌNH VUÔNG
# K FOLD CROSS VALIDATION
# KHUYẾN KHÍCH CHẠY PYTORCH
# CÔNG THỨC TÍNH BATCH SIZE PHÙ HỢP
# CHẠY RESNET
