import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import sympy as sym
from sympy import *
# from sympy import Eq

def binarize(img):
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2), -1)
    inverted_binary_img = ~binarized
    # print('000000000000')
    # print(inverted_binary_img)
    # print('11111111')
    return inverted_binary_img

data_dir = 'data'
batch_size = 32
img_height = 45
img_width = 45

# train_datagen = ImageDataGenerator(preprocessing_function=binarize)

# train_generator = train_datagen.flow_from_directory(
#         data_dir,
#         target_size=(img_height, img_width),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode="categorical",
#         seed=123)

# class_names = [k for k,v in train_generator.class_indices.items()]
#class_names = ['(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'cos', 'div', 'log', 'sin', 'tan', 'times', 'y']
#class_names = ['(',')','+','-','0','1','2','3','4','5','6','7','8','9','=','cos','div','log','sin','tan','times','y']
#class_names = ['+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9','times','y']
class_names = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times', 'y']
def getOverlap(a, b):
     return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours(img_path):
    # Given an image path, output bounding box coordinates of an image
    input_image = cv2.imread(img_path, 0) # Load a greyscale image
 
    # Make a copy to draw bounding box
    input_image_cpy = input_image.copy()

    # Convert the grayscale image to binary (image binarization opencv python), then invert
    binarized = cv2.adaptiveThreshold(input_image_cpy,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    inverted_binary_img = ~binarized

    # Detect contours
    # hierarchy variable contains information about the relationship between each contours
    contours_list, hierarchy = cv2.findContours(inverted_binary_img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE) # Find contours
    # Detect contours
    l = []
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        l.append([x, y, w, h])
    # Check whether any overlapping rectangles. We do this in a way such that we only compare each box once with all other boxes.
    lcopy = l.copy()
    keep = []
    while len(lcopy) != 0:
        curr_x, curr_y, curr_w, curr_h = lcopy.pop(0) # Look at next box
        if curr_w * curr_h < 20: # remove very small boxes
            continue
        throw = []
        for i, (x, y, w, h) in enumerate(lcopy):
            curr_interval = [curr_x, curr_x+curr_w]
            next_interval = [x, x+w]
            if getOverlap(curr_interval, next_interval) > 1 : # more than 3 pixels overlap, this is arbitrary
                # Merge the two intervals
                new_interval_x = [min(curr_x, x), max(curr_x+curr_w, x+w)]
                new_interval_y = [min(curr_y, y), max(curr_y+curr_h, y+h)]
                newx, neww = new_interval_x[0], new_interval_x[1] - new_interval_x[0]
                newy, newh = new_interval_y[0], new_interval_y[1] - new_interval_y[0]
                curr_x, curr_y, curr_w, curr_h = newx, newy, neww, newh
                throw.append(i) # Mark this box to throw away later, since it has now been merged with current box
        for ind in sorted(throw, reverse=True): # Sort in reverse order otherwise we will pop incorrectly
            lcopy.pop(ind)
        keep.append([curr_x, curr_y, curr_w, curr_h]) # Keep the current box we are comparing against
    return keep

def resize_pad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def put_double_asterisk(s):
    # Convert the string to a list
    lst = list(s)
    i = 0
    # Iterate over the list
    while i < len(lst)-1:
        if lst[i].isalpha():
            if lst[i+1].isdigit():
                # Insert the double asterisk
                lst.insert(i+1, '**')
                i += 1
        i += 1
    # Convert the list back to a string
    s_new = ''.join(lst)
    return s_new

def put_single_asterisk(s):
    lst = list(s)
    i=0
    while i < len(lst)-1:
        if lst[i].isdigit() and lst[i+1].isalpha():
            lst.insert(i+1,'*')
        i+=1
    s_new = ''.join(lst)
    return s_new 

    
# Load model
new_model = tf.keras.models.load_model('eqn-detect1 -model', compile=False)


def equation_solver_function(img_path):

    flag = 0
    print(img_path.split('\\')[-1])
    # print('10000000000000000000000')
    IMAGE = img_path.split('\\')[-1]#"test7.png"

    img_path = "static/"+IMAGE
    image_dir = "static/"

    if flag == 1:
        image_dir = "equation_images/"
        input_image = cv2.imread(img_path) 
        ret, bw_img = cv2.threshold(input_image,127,255,cv2.THRESH_BINARY)
        plt.imshow(bw_img)
        plt.show()
        cv2.imwrite("equation_images/savedimage.jpg", bw_img) 
        input_image_cpy = bw_img.copy()
        keep = detect_contours(image_dir+'savedimage.jpg')
        # print(len(keep))
        img_path = image_dir+'savedimage.jpg'
    else:
        print('============+++++++++++++++++')
        print(img_path)
        input_image = cv2.imread(img_path, 0) 
        input_image_cpy = input_image.copy()
        keep = detect_contours(image_dir+IMAGE)
        # print(len(keep))


    eqn_list = []
    # print("Below eqn list")
    # binarize the input image
    #IMAGE = "akshibig.jpg"
    #img_path = "equation_images/"+IMAGE
    input_image = cv2.imread(img_path, 0) 
    inverted_binary_img = binarize(input_image)
    # print("after in")
    for (x, y, w, h) in sorted(keep, key = lambda x: x[0]):
        # plt.imshow(inverted_binary_img[y:y+h, x:x+w])
        # plt.show()
        img = resize_pad(inverted_binary_img[y:y+h, x:x+w], (45, 45), 0) # We must use the binarized image to predict
        # plt.imshow(img)
        # plt.show()
        # print('1')
        first = tf.expand_dims(img, 0)
        # print('2')
        second = tf.expand_dims(first, -1)
        # print('3')
        predicted = new_model.predict(second)
        # print('4')
        max_arg = np.argmax(predicted)
        # print('5')
        pred_class = class_names[max_arg]
        # print('6')
        # print('7')
        if pred_class == "times":
            pred_class = "*"
        if pred_class == "div":
            pred_class = chr(47)
        eqn_list.append(pred_class)
        # print(pred_class)
        #plt.imshow(img)
        #plt.show()
    # print('8')
    eqn = "".join(eqn_list)
    # print(solve_equation(eqn))
    # print('9')
    print(eqn)
    equation = put_double_asterisk(eqn)
    equation = put_single_asterisk(equation)
    # result, equation = solve_equation(eqn)

    # return result, equation
    return equation




