from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
import numpy as np
import random
from dataset_parser.prepareData import VOCPalette
import time, math
import cv2
def load_data_org(path, mode=None):
    img = Image.open(path)
    if mode=="original":
        return img
    if mode=="label":
        y = np.array(img, dtype=np.int32)
        mask = y == 255
        y[mask] = 0
        #y = binarylab(y, size, 21)
        y = np.expand_dims(y, axis=-1)
        return y
    if mode=="data":
        if channel_num == 1:
            X = cv2.imread(path, 0)
            X = X.reshape((X.shape[0],X.shape[1],1))
        elif channel_num == 3:
            X = cv2.imread(path)
        #img.show()
        #X = np.expand_dims(X, axis=0)
        #X = preprocess_input(X)
        return X

x_data_gen_args = dict(shear_range=0.1,
                       zoom_range=0.1,
                       rotation_range=15,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       fill_mode='constant',
                       horizontal_flip=True)

y_data_gen_args = dict(shear_range=0.1,
                       zoom_range=0.1,
                       rotation_range=15,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       fill_mode='constant',
                       horizontal_flip=True)

GEN_NUM = 20
VAL_RATIO = 0.2
B_SIZE = 1
NB_CLASS = 5
channel_num = 1
input_file = './list1.txt'

img_path = './image/BUS_origin_one/'
label_path = './image/GT_tumor/'

gen_img_path = './gen/img/'
gen_lab_path = './gen/lab/'
gen_txt_path = './gen/'

with open(input_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

val_num = math.ceil(nb_data_img * VAL_RATIO)

random.seed(time.time)
#indices = [n for n in range(len(namesimg))]
#random.shuffle(indices)
#print(indices)

random.shuffle(namesimg)

palette = VOCPalette(nb_class=NB_CLASS)
# Make ImageDataGenerator.
x_data_gen = ImageDataGenerator(**x_data_gen_args)
y_data_gen = ImageDataGenerator(**y_data_gen_args)
f_train = open(gen_txt_path + 'train.txt', "a")
f_val = open(gen_txt_path + 'val.txt', "a")
f_test = open(gen_txt_path + 'test_tumor.txt', "a")

for i in range(nb_data_img):
    if i < val_num:
        f_test.writelines(namesimg[i] + "\n")

    Xpath = img_path + "{}.png".format(namesimg[i])
    Ypath = label_path + "{}.png".format(namesimg[i])
    print(Xpath)

    x = load_data_org(Xpath, 'data')
    y = load_data_org(Ypath, 'label')

    x=x.reshape((1,)+x.shape)
    y=y.reshape((1,)+y.shape)

    # Adapt ImageDataGenerator flow method for data augmentation.
    _ = np.zeros(B_SIZE)
    seed = random.randrange(1, 1000)

    x_tmp_gen = x_data_gen.flow(np.array(x), _, batch_size=B_SIZE, seed=seed)
    y_tmp_gen = y_data_gen.flow(np.array(y), _, batch_size=B_SIZE, seed=seed)

    # Finally, yield x, y data.
    for j in range(GEN_NUM):
        x_result, _ = next(x_tmp_gen)
        y_result, _ = next(y_tmp_gen)
        x_res = x_result[0]
        y_res = y_result[0]
        y_res = np.squeeze(y_res, axis=2)
        img = image.array_to_img(x_res)
        lab_arr = y_res.astype('uint8')
        lab = palette.genlabelpal(lab_arr)
        X_res_path = gen_img_path + "{}".format(namesimg[i]) + '_' + str(j) + ".png"
        Y_res_path = gen_lab_path + "{}".format(namesimg[i]) + '_' + str(j) + ".png"
        img.save(X_res_path)
        lab.save(Y_res_path)
        if i < val_num :
            f_val.writelines(namesimg[i] + '_' + str(j) + "\n")
        else:
            f_train.writelines(namesimg[i] + '_' + str(j) + "\n")

    #    img.show()
    #    lab.show()
f_train.close()
f_val.close()
f_test.close()
