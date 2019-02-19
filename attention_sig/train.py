from __future__ import print_function

import os
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck

from model.unet import unet,net_from_json
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from dataset_parser.generator import data_generator_dir

# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet'],
                    help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg

TRAIN_BATCH = 5
VAL_BATCH = 1
epochs = 80
resume_training = False

path_to_train = './train.txt'
path_to_val = './val.txt'
path_to_img = './image/BUS/'
path_to_label = './image/GT_tumor/'
img_width = 256
img_height = 256
nb_class = 5
channels = 1
f_loss = open("loss_1.txt","a")

with open(path_to_train,"r") as f:
    ls = f.readlines()
namestrain = [l.rstrip('\n') for l in ls]
nb_data_train = len(namestrain)

with open(path_to_val,"r") as f:
    ls = f.readlines()
namesval = [l.rstrip('\n') for l in ls]
nb_data_val = len(namesval)

if os.path.exists(model_name + '_model_struct.json') == True:
    # create model from JSON
    print('Reading network...\n')
    model = net_from_json(model_name + '_model_struct.json', lr_init, lr_decay)
else:
    # Create model to train
    print('Creating network...\n')
    if model_name == "fcn":
        model = fcn_8s(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                       lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
    elif model_name == "unet":
        model = unet(input_shape=(img_height, img_width, channels), num_classes=nb_class,
                     lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
    elif model_name == "pspnet":
        model = pspnet50(input_shape=(img_height, img_width, channels), num_classes=nb_class, lr_init=lr_init, lr_decay=lr_decay)


# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_model_checkpoint_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
train_check = TrainCheck(output_path='./img', model_name=model_name, img_shape=(img_height,img_width),nb_class=nb_class)
#early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

#load weights if needed
if resume_training:
    print('Resume training...\n')
    model.load_weights(model_name + '_model_checkpoint_weight.h5')
else:
    print('New training...\n')

# training
history = model.fit_generator(data_generator_dir(namestrain, path_to_img, path_to_label,(img_height, img_width, channels), nb_class, TRAIN_BATCH, 'train'),
                              steps_per_epoch=nb_data_train // TRAIN_BATCH,
                              validation_data=data_generator_dir(namesval, path_to_img, path_to_label, (img_height, img_width, channels), nb_class, VAL_BATCH, 'val'),
                              validation_steps=nb_data_val // VAL_BATCH,
                              callbacks=[checkpoint,train_check],
                              epochs=epochs,
                              verbose=1)


# serialize model weigths to h5
f_loss.write(str(history.history))
# serialize model weigths to h5
model.save_weights(model_name + '_model_weight.h5')
f_loss.close()

