# USAGE
# python video.py
import tensorflow as tf

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from matplotlib.patches import Rectangle
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
import os

'''
BEFORE USING MAKE SURE TO DOWNLOAD THE MODEL AT https://www.dropbox.com/s/5rsqhw9emxy9ypd/MODEL_CLEAN_V3.hdf5?dl=0

ONCE THE MODEL HAS BEEN DONWLOADED LOAD IT IN THE FOLLOWING VARIABLE
'''

MODEL = 'Path_to_model' 

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction= 0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
detector = MTCNN()


def mask_detect(img):
    """
    This function takes the path of an image as input.
    The image is run through the MTCNN face detector; the detected
    faces are then run through our neural network to detect the
    presence of a face mask and the result is given as an
    output image.
    """
    # print(img.shape)
    image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    w, h, _ = image.shape
    if w < 224 and h < 224:
        print("Image resolution too low to be analyzed!")
        return

    locs = []
    logit = []

    # pass the image through the MTCNN and obtain the face detections
    faces = detector.detect_faces(image)

    face_counter = 0
    # loop over the detection

    for face in faces:

        # extract the confidence associated with the detection
        confidence = face["confidence"]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.9:
            face_counter += 1
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = face["box"]
            (startX, startY, width, height) = box

            # extract the face, resize it to 224x224
            face = image[max(0, startY):startY+height,
                             max(0, startX):startX+width]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            logit.append(model.predict(face))
            locs.append((startX, startY, startX+width, startY+height))

    return (locs, logit)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2_imshow(image)


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")

model = load_model(MODEL)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
     
	(locs, preds) = mask_detect(frame)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if pred > 0.5 else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

	

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
