from keras.layers import Layer, concatenate
import tensorflow as tf
from keras import initializers
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# tfb = tfp.distributions.bijectors

# tfd = tf.contrib.distributions


class FuzzyLayer(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[2]
        self.output_dim = output_dim
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale_1 = self.add_weight(name='v1', shape=(1,self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean_1 = self.add_weight(name='v2', shape=(1,self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform',
                                    trainable=True)
        self.scale_2 = self.add_weight(name='v3', shape=(1,self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean_2 = self.add_weight(name='v4', shape=(1,self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform',
                                    trainable=True)
        self.scale_3 = self.add_weight(name='v5', shape=(1,self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean_3 = self.add_weight(name='v6', shape=(1,self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform',
                                    trainable=True)
        self.scale_4 = self.add_weight(name='v7', shape=(1,self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean_4 = self.add_weight(name='v8', shape=(1,self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform',
                                    trainable=True)
        self.scale_5 = self.add_weight(name='v9', shape=(1,self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean_5 = self.add_weight(name='v10', shape=(1,self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform',
                                    trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):
        self.scale_temp = concatenate(
            [self.scale_1, self.scale_2, self.scale_3, self.scale_4, self.scale_5], axis=0)
        self.mean_temp = concatenate([self.mean_1, self.mean_2, self.mean_3, self.mean_4,
                                      self.mean_5], axis=0)
        output = []
        for i in range(self.class_num):
            x1 = tf.subtract(x, self.mean_temp[i])
            x2 = tf.square(x1)
            x3 = tf.divide(x2, self.scale_temp[i])
            x4 = tf.multiply(-0.5, x3)
            gauss = tf.exp(x4)

            output.append(gauss[:,:,:,0])

        output = tf.convert_to_tensor(output)
        output = tf.transpose(output, perm=[1, 2, 3, 0])
        sum1 = tf.reduce_sum(output, axis=3)
        sum1 = tf.reshape(sum1, [-1, self.img_height, self.img_width, 1])
        sum1 = tf.tile(sum1, multiples=[1, 1, 1, self.class_num])
        output = tf.divide(output, sum1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.class_num)
        return output_shape


class FuzzyLayer_sigmoid(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[2]
        self.output_dim = output_dim
        super(FuzzyLayer_sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='v1', shape=(self.class_num, 1, self.channel_num), initializer='uniform',
                                    trainable=True)
        self.b = self.add_weight(name='v2', shape=(self.class_num, 1, self.channel_num), initializer='uniform',
                                    trainable=True)
        super(FuzzyLayer_sigmoid, self).build(input_shape)

    def call(self, x):
        self.a_temp = tf.tile(self.a, multiples=[1, self.img_height * self.img_width, 1])
        self.b_temp = tf.tile(self.b, multiples=[1, self.img_height * self.img_width, 1])
        x = tf.reshape(x, [-1, self.img_height * self.img_width, self.channel_num])
        output = []
        for i in range(self.class_num):
            x1 = tf.subtract(x, self.b_temp[i])
            x2 = tf.multiply(self.a_temp[i], x1)
            gauss = tf.nn.sigmoid(x2)
            output.append(gauss[:,:,0])

        output = tf.convert_to_tensor(output)
        output = tf.transpose(output, perm=[1, 2, 0])
        output = tf.reshape(output, [-1, self.img_height, self.img_width, self.class_num])
        output = tf.nn.l2_normalize(output, dim=3)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.class_num)
        return output_shape

class myMultiLayer(Layer):
    def __init__(self, input_dim, output_dim, fuzzyInput, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[2]
        self.output_dim = output_dim
        self.gauss = fuzzyInput
        super(myMultiLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(myMultiLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = []
        for i in range(self.class_num):
            gauss_tensor = tf.reshape(tf.slice(self.gauss, [0, 0, 0, i], [-1, -1, -1, 1]),
                                      [-1, self.img_height * self.img_width])
            for j in range(self.channel_num):
                temp = tf.multiply(gauss_tensor, tf.reshape(tf.slice(x, [0, 0, 0, j], [-1, -1, -1, 1]),
                                                            [-1, self.img_height * self.img_width]))
                output.append(temp)

        output = tf.reshape(tf.convert_to_tensor(output),
                            [-1, self.img_height, self.img_width, self.class_num * self.channel_num])

        return tf.concat([x, output], 3)    #concatenate([x, output], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = (
        input_shape[0], self.output_dim[0], self.output_dim[1], self.channel_num + self.class_num * self.channel_num)
        return output_shape


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
        gauss = []
        output = []
        for i in range(self.class_num):
            mvn = tfd.MultivariateNormalTriL(
                loc=self.mean_temp[i],
                scale_tril=self.scale_temp[i])
            gauss_tensor = mvn.prob(x)
            gauss.append(gauss_tensor)

        gg = tf.convert_to_tensor(gauss)
        gauss = tf.reshape(tf.convert_to_tensor(gauss), [-1, self.img_height, self.img_width, self.class_num])
        gauss = tf.nn.l2_normalize(gauss, dim=3)

        for i in range(self.class_num):
            gauss_tensor = tf.reshape(tf.slice(gauss, [0, 0, 0, i], [-1, -1, -1, 1]),
                                      [-1, self.img_height * self.img_width])
            for j in range(self.channel_num):
                temp = tf.multiply(gauss_tensor, tf.reshape(tf.slice(x, [0, 0, j], [-1, -1, 1]),
                                                            [-1, self.img_height * self.img_width]))
                output.append(temp)

        output = tf.reshape(tf.convert_to_tensor(output),
                            [-1, self.img_height, self.img_width, self.class_num * self.channel_num])
        x = tf.reshape(x, [-1, self.img_height, self.img_width, self.channel_num])

        return tf.concat([gauss, x, output], 3)    #concatenate([gauss, x, output], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1],
                        self.class_num + self.channel_num + self.class_num * self.channel_num)
        return output_shape


class FuzzyMergeLayer(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[2]
        self.output_dim = output_dim
        super(FuzzyMergeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='v1', shape=(self.class_num, 1, self.channel_num, self.channel_num),
                                     initializer='uniform', trainable=True)
        self.mean = self.add_weight(name='v2', shape=(self.class_num, 1, self.channel_num), initializer='uniform',
                                    trainable=True)
        super(FuzzyMergeLayer, self).build(input_shape)

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
            output.append(gauss)

        output = tf.reshape(tf.convert_to_tensor(output), [-1, self.img_height, self.img_width, self.class_num])
        output = tf.nn.l2_normalize(output, dim=3)
        x = tf.reshape(x, [-1, self.img_height, self.img_width, self.channel_num])
        return tf.concat([x, output], 3)    #concatenate([x, output], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.channel_num + self.class_num)
        return output_shape
