from keras.models import Model,model_from_json
from keras.layers import Input, multiply, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Lambda
from model.gaussLayer import FuzzyLayer_sigmoid
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
def fuzzyfunc(x):
    fuzzy_W_tensor1 = tf.nn.relu(tf.subtract(0.5, x))
    fuzzy_W_tensor2 = tf.nn.relu(tf.subtract(x, 0.5))
    # map: 0--0.5 => 0--0.5; 0.5--1 => 0.5--0
    fuzzy_W_tensor = tf.subtract(0.5, tf.add(fuzzy_W_tensor1, fuzzy_W_tensor2))
    # map to 0--1
    fuzzy_W_tensor = tf.multiply(2.0, fuzzy_W_tensor)
    fuzzy_W_tensor = tf.subtract(1.0, fuzzy_W_tensor)
    return fuzzy_W_tensor
def unet(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    fuzzyLayer = FuzzyLayer_sigmoid((input_shape[0], input_shape[1], 1), (input_shape[0], input_shape[1], num_classes))(img_input)
    fuzzy_W_tensor = Lambda(lambda x: fuzzyfunc(x))(fuzzyLayer)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)

    fuzzy_W_tensor1 = Lambda(lambda x: tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1]))(fuzzy_W_tensor)
    fuzzy_W_tensor2 = Lambda(lambda x: tf.slice(x, [0, 0, 0, 1], [-1, -1, -1, 1]))(fuzzy_W_tensor)
    fuzzy_W_tensor3 = Lambda(lambda x: tf.slice(x, [0, 0, 0, 2], [-1, -1, -1, 1]))(fuzzy_W_tensor)
    fuzzy_W_tensor4 = Lambda(lambda x: tf.slice(x, [0, 0, 0, 3], [-1, -1, -1, 1]))(fuzzy_W_tensor)
    fuzzy_W_tensor5 = Lambda(lambda x: tf.slice(x, [0, 0, 0, 4], [-1, -1, -1, 1]))(fuzzy_W_tensor)
    productlayer1 = multiply([x, fuzzy_W_tensor1])
    productlayer2 = multiply([x, fuzzy_W_tensor2])
    productlayer3 = multiply([x, fuzzy_W_tensor3])
    productlayer4 = multiply([x, fuzzy_W_tensor4])
    productlayer5 = multiply([x, fuzzy_W_tensor5])

    x = concatenate([x, productlayer1, productlayer2, productlayer3, productlayer4, productlayer5], axis=3)
    x = Conv2D(64, (1, 1), padding='same', name='block1_conv3')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)

    x = MaxPooling2D()(block_4_out)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model

def net_from_json(path, lr_init, lr_decay):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),loss='categorical_crossentropy', metrics=[dice_coef])

    return model

from keras.layers import Layer
import tensorflow as tf

tfd = tf.contrib.distributions

class FuzzyMultiLayer(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[2]
        self.output_dim = output_dim
        super(FuzzyMultiLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='v1', shape=(self.class_num, 1, self.channel_num, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2', shape=(self.class_num, 1, self.channel_num), initializer='uniform',
                                    trainable=True)
        super(FuzzyMultiLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.scale_temp = tf.tile(self.scale, multiples=[1, self.img_height * self.img_width, 1, 1])
        self.mean_temp = tf.tile(self.mean, multiples=[1, self.img_height * self.img_width, 1])
        x = tf.reshape(x, [-1, self.img_height * self.img_width, self.channel_num])
        output = []
        for i in range(self.class_num):
            mvn = tfd.MultivariateNormalTriL(
                loc=self.mean_temp[i],
                scale_tril=self.scale_temp[i])
            gauss = mvn.prob(x)
            for j in range(self.channel_num):
                temp = tf.multiply(gauss, tf.reshape(tf.slice(x, [0, 0, j], [-1, -1, 1]),
                                                     [-1, self.img_height * self.img_width]))
                output.append(temp)
        output = tf.reshape(tf.convert_to_tensor(output),
                            [-1, self.img_height, self.img_width, self.class_num * self.channel_num])
        x = tf.reshape(x, [-1, self.img_height, self.img_width, self.channel_num])
        return concatenate([x, output], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.class_num * self.channel_num + self.channel_num)
        return output_shape
