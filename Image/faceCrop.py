import numpy as np
import cv2
import scipy.misc
import os
import dlib
import imutils
from imutils import face_utils

face_cascade = cv2.CascadeClassifier('C:\\Users\DIRL\Downloads\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\DIRL\Downloads\opencv-master\opencv-master\data\haarcascades\haarcascade_eye.xml')

inpit_dir = 'C:\\Users\DIRL\Desktop\\facecrop\origin'
output_dir = 'C:\\Users\DIRL\Desktop\\facecrop\\faceCroped'
shape_pred_path = 'C:\\Users\DIRL\Downloads\\facial-landmarks\\facial-landmarks\shape_predictor_68_face_landmarks.dat'


def face_crop(inDir_path, outDir_path, resized_size, faceBorder_scale):

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_pred_path)


    file_list = os.listdir(inDir_path)
    for file in file_list:

        *file_name, file_format = file.split('.')
        print(file)

        # load the input image, resize it, and convert it to grayscale
        image = scipy.misc.imread(inDir_path+'\\'+file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            print(i)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # shape = predictor(gray, rect)
            # shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x+int(w/2)
            center_y = y+int(h/2)
            width    = w
            height   = h

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            # for (x, y) in shape:
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            scaled_w = int(faceBorder_scale*width)
            scaled_h = int(faceBorder_scale*height)

            left = center_x - int(scaled_w/2)
            top  = center_y - int(scaled_h/2)

            if left < 0:
                left = 0
            if top < 0:
                top = 0

            crop_img = image[top:top+scaled_h, left:left+scaled_w]

        # resize
        crop_width = crop_img.shape[1]
        crop_height = crop_img.shape[0]

        if crop_width > crop_height:
            scale = resized_size/crop_height
            crop_img = scipy.misc.imresize(crop_img, scale)
            center_x = int(scale * center_x)
            center_y = int(scale * center_y)
            scaled_w = int(scale * width)
            scaled_h = int(scale * height)
            # cv2.rectangle(crop_img, (int(scale*x), int(scale*y)), (int(scale*x) + scaled_w, int(scale*y) + scaled_h), (0, 0, 255), 2)
            crop_img = crop_img[:, int((crop_img.shape[1]-resized_size)/2): int((crop_img.shape[1]-resized_size)/2)+ resized_size]
        else:
            scale = resized_size/crop_width
            crop_img = scipy.misc.imresize(crop_img, scale)
            left = int(scale*x)
            top = int(scale*y)
            scaled_w = int(scale * width)
            scaled_h = int(scale * height)
            border_toTop = top
            border_toButtom = crop_img.shape[0] - scaled_h - top

            # cv2.rectangle(crop_img, (int(scale*x), int(scale*y)), (int(scale*x) + scaled_w, int(scale*y) + scaled_h), (0, 0, 255), 2)

            start_from = int((crop_img.shape[0]-resized_size)*border_toTop/(border_toTop+border_toButtom))
            crop_img = crop_img[ start_from : start_from + resized_size, :]

        scipy.misc.imsave(outDir_path+'\\'+file_name[0]+'_s190.png', crop_img)


face_crop(inpit_dir, output_dir, 190, 5)
