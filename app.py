import imageai
from imageai.Detection import ObjectDetection
import hyper as hp
import cv2
import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
import keras.backend as K

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST','GET'])
def hello():
    K.clear_session()

    if request.method == 'POST':
        detector = ObjectDetection()
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath("yolo-tiny.h5")
        detector.loadModel()
        algo = request.form['algo']
        if 'file_input' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file_input']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_data = file.stream.read()
                nparr = np.fromstring(file_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                detect_img, result = detector.detectObjectsFromImage(input_image=img,
                                                input_type="array",
                                                output_type="array",
                                                minimum_percentage_probability=80)
                K.clear_session()
                detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('detect_img.jpg', detect_img)
                img_str = cv2.imencode('.jpg', detect_img)[1].tostring()
                encoded = base64.b64encode(img_str).decode("utf-8")
                mime = "image/jpg;"
                out_image = f"data:{mime}base64,{encoded}"
                return render_template('result.html', out_image=out_image)
        else:
                return "File extension not supported"
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = "key_key"
    app.run(debug=False)