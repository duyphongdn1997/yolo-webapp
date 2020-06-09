from imageai.Detection import ObjectDetection
import os
import cv2
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()
img = cv2.imread('car.jpg')

detect_img, result = detector.detectObjectsFromImage(input_image=img,
                                                input_type="array",
                                                output_type="array",
                                                minimum_percentage_probability=30)
print(detect_img)

print(result)