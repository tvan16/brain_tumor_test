import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('Braintumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"

#hàm thực hiện dự đoán trên hình ảnh
def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((256, 256))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_index = np.argmax(result, axis=1)[0]
    return class_index

#tạo route cho máy chủ
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST': #Chỉ xử lý khi phương thức HTTP là POST
        f = request.files['file'] # Lấy file hình ảnh từ form mà người dùng tải lên
 
        basepath = os.path.dirname(__file__) #Lấy đường dẫn của thư mục hiện tại.
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)) #Tạo đường dẫn đầy đủ để lưu file tải lên trong thư mục uploads.
        f.save(file_path) # Lưu file hình ảnh vào đường dẫn được chỉ định.
        value=getResult(file_path) # Thực hiện dự đoán với hình ảnh đã tải lên.
        result=get_className(value) #Chuyển kết quả dự đoán thành chuỗi mô tả.
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)