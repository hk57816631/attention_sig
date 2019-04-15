from __future__ import print_function
import os
import argparse
from PIL import Image
import numpy as np
from dataset_parser.prepareData import VOCPalette

from model.unet import unet,net_from_json
from model.fcn import fcn_8s
from model.pspnet import pspnet50

import matplotlib.pyplot as plt

labelcaption = ['background','tumor','fat','mammary','muscle','bottle','bus','car','cat',
                'chair','cow','Dining table','dog','horse','Motor bike','person','Potted plant',
                'sheep','sofa','train','monitor']
def result_map_to_img(res_map):
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2).astype('uint8')

    return argmax_idx

def findObject(labelimg):
    counts = np.zeros(20,dtype=np.int32)
    str_obj = ''
    for i in range(20):
        counts[i] = np.sum(labelimg == i+1)
        if counts[i] > 500 :
            str_obj = str_obj + labelcaption[i+1]+ ' '
    return str_obj

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-P", "--img_path", required=False, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path
vgg_path = None

img_path = './image/BUS/'
label_path = './image/GT_tumor/'
test_file = './val.txt'

img_width = 256
img_height = 256
nb_class = 5
channels = 1

if os.path.exists(model_name + '_model_struct.json') == True:
    # create model from JSON
    print('Reading network...\n')
    model = net_from_json(model_name + '_model_struct.json', 1e-3, 5e-4)
else:
    # Create model to train
    print('Creating network...\n')
    if model_name == "fcn":
        model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                       lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
    elif model_name == "unet":
        model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                     lr_init=1e-3, lr_decay=5e-4, vgg_weight_path=vgg_path)
    elif model_name == "pspnet":
        model = pspnet50(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=1e-3, lr_decay=5e-4)

    # serialize model structure to JSON
    model_json = model.to_json()
    # with open(model_name + '_model_struct.json', 'w') as json_file:
    #    json_file.write(model_json)
try:
    model.load_weights(model_name + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

palette = VOCPalette(nb_class=nb_class)

with open(test_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

for i in range(nb_data_img):
    Xpath = img_path + "{}.png".format(namesimg[i])
    Ypath = label_path + "{}.png".format(namesimg[i])

    print(Xpath)

    # imgorg = Image.open(Xpath).convert('RGB')
    imgorg = Image.open(Xpath)
    imglab = Image.open(Ypath)
    img = imgorg.resize((img_width, img_height), Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = img_arr / 127.5 - 1
    img_arr = np.expand_dims(img_arr, 0)
    img_arr = img_arr.reshape((1, img_arr.shape[1], img_arr.shape[2], 1))
    pred = model.predict(img_arr)
    res = result_map_to_img(pred[0])
    PIL_img_pal = palette.genlabelpal(res)
    PIL_img_pal = PIL_img_pal.resize((imgorg.size[0], imgorg.size[1]), Image.ANTIALIAS)

    obj = findObject(np.array(PIL_img_pal))

    plt.ion()
    plt.figure('Unet test')  
    plt.suptitle(Xpath) 
    plt.subplot(1, 3, 1), plt.title('org')
    plt.imshow(imgorg), plt.axis('off')
    plt.subplot(1, 3, 2), plt.title(obj)
    plt.imshow(PIL_img_pal), plt.axis('off')
    plt.subplot(1, 3, 3), plt.title('label')
    plt.imshow(imglab), plt.axis('off')

    plt.show()
    str = input("press any key...")
    if str == 'q' or str == 'Q':
        break
    plt.close(1)
plt.close('all')


