# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import argparse
# import text_detection
from PIL import Image, ImageEnhance


#preprocess
def preprocess_image(image_path, resize = True):

    im = Image.open(image_path)
    # decrease brightness
    enhancer = ImageEnhance.Brightness(im)
    factor = 0.2
    bright = enhancer.enhance(factor)

    # increase contrast
    enhancer = ImageEnhance.Contrast(bright)
    factor = 6.0
    img = enhancer.enhance(factor)
    image = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

    # resize image
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = new_image /255
    if resize:
        new_image = cv2.resize(new_image, (227,227))

    return new_image
# check the background color of image
def bg_color(plate_image):

    # convert to binary image
    plate_image = cv2.convertScaleAbs(plate_image, alpha= (255.0))
    gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 188, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # find contours of binary image
    _ ,cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # count black pixels and white pixels of countours
    black_count = 0
    white_count = 0

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=24: # Only select contour with defined ratio
            if h/plate_image.shape[0] >= 0.5 and h/plate_image.shape[0] < 1: # Select contour which has the height larger than 50% of the plate
                
                if np.any(binary[y, x:x+w]==0): # count black pixels of top 
                    black_count += 1
                elif np.any(binary[y, x:x+w]==255) : # count white pixels of top
                    white_count += 1
                elif np.any(binary[y+h, x:x+w]==0): # count black pixels of bottom 
                    black_count += 1
                elif np.any(binary[y+h, x:x+w]==255): # count white pixels of bottom 
                    white_count += 1
                elif np.all(binary[y:y+h, x] == 0): # count black pixels of left 
                    black_count +=1
                elif np.all(binary[y:y+h, x] == 255): # count white pixels of left 
                    white_count +=1
                elif np.all(binary[y:y+h, x+w]== 0): # count black pixels of right 
                    black_count += 1
                elif np.all(binary[y:y+h, x+w]== 255): # count white pixels of right 
                    white_count += 1
            
    if black_count > white_count:
        result = 'black'
    else:
        result = 'white'

    return result, gray

def preprocess(plate_image):

    bgColor, gray = bg_color(plate_image)
    print('Background color: '+bgColor)

    blur = cv2.GaussianBlur(gray,(1,3),0)

    # convert background color to black
    if bgColor == 'black':
        binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)

    # cv2.imshow('thre_mor',thre_mor)
    # cv2.waitKey()
    return thre_mor

#Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts, reverse= False):
    boundingboxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingboxes) = zip(*sorted(zip(cnts, boundingboxes), key= lambda b:b[1][0], reverse= False))
    return cnts

def crop_character():
    binary = preprocess(plate_image)
    _ ,cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plate_image to draw bounding box
    test_roi = plate_image.copy()
    data = binary.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # Define the standard size 
    digit_w = 30
    digit_h = 60
    
    box_w = []

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        box_w.append(w)
        ratio = h/w
        if 1<=ratio<=25: # Only select contour with defined ratio
            if h/plate_image.shape[0] >= 0.5 and h/plate_image.shape[0] < 1: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                # Sperate number and gibe prediction
                curr_num = binary[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cv2.imshow('cur',curr_num)
                # cv2.waitKey()

                # resize image of I or 1
                if ratio >= 10:
                    curr_num = cv2.resize(curr_num,dsize=(60//int(ratio),60))

                    #Creating a standard image with NUMPY  
                    f = np.zeros((60,30),np.uint8)

                    #Getting the centering position
                    ax = (30 - curr_num.shape[1])//2

                    #Pasting the 'image' in a centering position
                    f[0:60,ax:ax+curr_num.shape[1]] = curr_num

                    # cv2.imshow("IMG",f)
                    # cv2.waitKey(0)
                    curr_num = f

                crop_characters.append(curr_num)

   
    # cv2.imshow('crop',test_roi)
    # cv2.waitKey()

    print("Detect {} letters...".format(len(crop_characters)))
    return crop_characters

# Load model architecture, weight and labels
def load_model():
    json_file = open('/home/medisys/Desktop/pyPro/ocr/ocr_August/MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("/home/medisys/Desktop/pyPro/ocr/ocr_August/License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('/home/medisys/Desktop/pyPro/ocr/ocr_August/license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")
    return model, labels

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

# get result
def get_string():
    fig = plt.figure(figsize=(15,3))
    crop_characters = crop_character()
    cols = len(crop_characters)
    grid = gridspec.GridSpec(ncols=cols,nrows=1)

    

    final_string = ''
    for i,character in enumerate(crop_characters):
        fig.add_subplot(grid[i])
        model, labels = load_model()
        title = np.array2string(predict_from_model(character,model,labels))
        plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        plt.axis('off')
        plt.imshow(character,cmap='gray') 

        
        cv2.imwrite(character_path+'/'+title+'_%d.jpg'%(i),character)     
    # plt.show()

    return final_string

# plate_path = text_detection.get_plate_crop()
plate_path = '/home/medisys/Desktop/pyPro/ocr/ocr_August/images/August_images/17'
path = plate_path+'.jpg'

base = os.path.basename(path)
number = os.path.splitext(base)[0]
character_path = '/home/medisys/Desktop/pyPro/ocr/ocr_August/images/characters/'+number
os.mkdir(character_path)


strings = []
for file_name in glob.iglob(plate_path+'/*.jpg', recursive=True):
    plate_image = preprocess_image(file_name)
    string = get_string()
    strings.append(string)

# strings = []
# img_path = '/home/medisys/Desktop/pyPro/ocr/ocr/images/reality/5/plate2.jpg'
# plate_image = preprocess_image(img_path)
# string = get_string()
# strings.append(string)

def get_list():
    return strings

print(get_list())

