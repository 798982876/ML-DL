# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

#########################################################
# construct the argument parser and parse the arguments
#--image: The path to the input image containing faces for inference
#--model: The path to the face mask detector model that we trained earlier in this tutorial
############################################################
def predict(img,model):
	# args = {'model': 'mask_detector7.model'}
	# print(args)
	# print("[INFO] loading face mask detector model...")
	model = load_model(model)
	image = cv2.imread(img)
	# print("[INFO] computing face detections...")
	face = image[:,:]
	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (150, 150))
	face = img_to_array(face)
	face = preprocess_input(face)
	face = np.expand_dims(face, axis=0)
	# pass the face through the model to determine if the face
	# has a mask or not
	(mask, withoutMask) = model.predict(face)[0]
	# print(mask, withoutMask)

	if mask > withoutMask:
		return 0
	else:
		return 1

args = {'dataset': 'dataset','model': 'mask_detector7.model'}
imagePaths = list(paths.list_images(args["dataset"]))
# print(imagePaths)
labels = []
real = []
predicts = []
for imgPath in imagePaths:
	# print(imgPath)
	# extract the class label from the filename
	label = imgPath.split(os.path.sep)[-2]
	if label == "with_mask":
		real.append(0)
	else:
		real.append(1)
	# print("label",label)
	# print(label)
	# load the input image (224x224) and preprocess it
	# img = cv2.imread(imgPath)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# print(img)
	predresult = predict(imgPath, args["model"])
	predicts.append(predresult)
	# print("result",result)
	labels.append(label)
# convert the data and labels to NumPy arrays
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("labels",labels)

#预测结果评价
print(classification_report(real, predicts,
	target_names=lb.classes_))
