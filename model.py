import tensorflow as tf
import os
import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, Activation, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Layer, InputSpec

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
# from keras.engine import Layer, InputSpec
from tensorflow.keras import backend as K


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=1,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Wavenet_blocks(Model):
    def __init__(self, input_size, num_channel, num_filters, kernel_size, num_residual_blocks):
        super(Wavenet_blocks, self).__init__()
        self.input_size = input_size
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_residual_blocks = num_residual_blocks
        self.wavenet_blocks = self.build_wavenet_model(self.input_size, self.num_channel, self.num_filters, self.kernel_size, self.num_residual_blocks)

    def get_config(self):
        config = {
            'input_size': self.input_size,
            'num_channel': self.num_channel,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_residual_blocks': self.num_residual_blocks
        }
        base_config = super(Wavenet_blocks, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def call(self, inputs):
        output = self.wavenet_blocks(inputs)
        return output

    def wavenet_residual_conv1d(self, num_channel, num_filters, kernel_size, dilation_rate):

        def build_residual_block(l_input):
            # dilated convolution
            l_norm = GroupNormalization(groups=1)(l_input)
            l_dilated_conv1d = Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='relu', groups=4)(l_norm)

            # sigmoid gate
            l_sigmoid_norm = GroupNormalization(groups=1)(l_dilated_conv1d)
            l_sigmoid_conv1d = Conv1D(num_filters, kernel_size=1, activation='sigmoid', groups=4)(l_sigmoid_norm)

            # tanh gate
            l_tanh_norm = GroupNormalization(groups=1)(l_dilated_conv1d)
            l_tanh_conv1d = Conv1D(num_filters, kernel_size=1, activation='tanh', groups=4)(l_tanh_norm)

            # multiply (sigmoid * tanh)
            l_mul = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])

            # skip connection
            l_skip = GroupNormalization(groups=1)(l_mul)
            l_skip_connection = Conv1D(num_channel, kernel_size=1, groups=4)(l_skip)

            # residual
            l_res = GroupNormalization(groups=1)(l_mul)
            l_residual = Conv1D(num_filters, kernel_size=1, activation='relu', groups=4)(l_res)
            l_residual = Add()([l_input, l_residual])

            return l_residual, l_skip_connection

        return build_residual_block

    def build_wavenet_model(self, input_size, num_channel, num_filters, kernel_size, num_residual_blocks):
        # input layer
        l_input = Input(batch_shape=(None, input_size, num_channel,))

        # channel expansion
        l_norm = GroupNormalization(groups=1)(l_input)
        l_stack_conv1d = Conv1D(num_filters, kernel_size=1, groups=4)(l_norm)

        l_skip_connections = []
        l_new_skip = []

        # repeat for each residual block
        for i in range(num_residual_blocks):
            l_stack_conv1d, l_skip_connection = self.wavenet_residual_conv1d(num_channel, num_filters, kernel_size, 2 ** i)(l_stack_conv1d)
            l_skip_connections.append(l_skip_connection)
            l_skip = tf.expand_dims(l_skip_connection, -1)
            l_new_skip.append(l_skip)

        l_concatenate = Concatenate()(l_new_skip)

        model = Model(inputs=l_input, outputs=l_concatenate)
        return model


class Wavenet_reconstruction(Model):
    def __init__(self, input_size, num_channel, kernel_size, num_blocks):
        super(Wavenet_reconstruction, self).__init__()
        self.input_size = input_size
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.receptive_field = self.get_receptive_field(self.kernel_size, self.num_blocks)

        ########### kyeonji ##########
        self.krelu = Activation('relu')
        self.kgroup1 = GroupNormalization(groups=1)
        self.kconv1d1 = Conv1D(self.num_channel, 1, activation='relu', padding='causal', groups=4)
        self.kgroup2 = GroupNormalization(groups=1)
        self.kconv1d2 = Conv1D(self.num_channel, 1, padding='causal', groups=4)
        ##############################

        # self.conv1d = Conv1D(self.num_channel, kernel_size=self.kernel_size, dilation_rate=2 ** self.num_blocks, padding='causal', groups=4)

    def get_config(self):
        config = {
            'input_size': self.input_size,
            'num_channel': self.num_channel,
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks
        }
        base_config = super(Wavenet_reconstruction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        # add each layer
        add1 = tf.reduce_sum(inputs, axis=-1)

        ########## kyeonji ###########
        relu = self.krelu(add1)
        relu1 = self.kgroup1(relu)
        l1_conv1d = self.kconv1d1(relu1)
        l1_conv1d1 = self.kgroup2(l1_conv1d)
        l2_conv1d = self.kconv1d2(l1_conv1d1)
        output = l2_conv1d[:, self.receptive_field - 1:-1, :]
        ##############################

        # # get output layer
        # conv1d = self.conv1d(add1)
        # output = conv1d[:, self.receptive_field - 1:-1, :]

        return output

    def get_receptive_field(self, kernel_size, num_blocks):
        receptive_field = 1
        for ii in range(num_blocks + 1):
            receptive_field = receptive_field * 2 + (kernel_size - 2)

        return receptive_field


class Resnet18(Model):
    def __init__(self, n_time, n_freq, n_channel, n_output):
        super(Resnet18, self).__init__()
        self.n_time = n_time
        self.n_freq = n_freq
        self.n_channel = n_channel
        self.n_output = n_output

        # self.filters = [64, 128, 256, 512]
        self.filters = [16, 32, 64, 128]

        self.network = self.build_resnet18(self.n_time, self.n_freq, self.n_channel)

    def get_config(self):
        config = {
            'n_time': self.n_time,
            'n_freq': self.n_freq,
            'n_channel': self.n_channel,
            'n_output': self.n_output
        }
        base_config = super(Resnet18, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build_resnet18(self, n_time, n_freq, n_channel):
        inputs = Input(shape=(n_time, n_freq, n_channel))

        conv1 = Conv2D(self.filters[0], 7, strides=2, padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        activation1 = Activation('relu')(bn1)
        mp1 = MaxPooling2D((3, 3), strides=2, padding='same')(activation1)

        rb1 = self.build_residual_blocks(mp1.shape[1], mp1.shape[2], mp1.shape[3], self.filters[0], 3)(mp1)
        rb2 = self.build_residual_blocks(rb1.shape[1], rb1.shape[2], rb1.shape[3], self.filters[0], 3)(rb1)

        rb3 = self.build_blocks(rb2.shape[1], rb2.shape[2], rb2.shape[3], self.filters[1], 3)(rb2)
        rb4 = self.build_residual_blocks(rb3.shape[1], rb3.shape[2], rb3.shape[3], self.filters[1], 3)(rb3)

        rb5 = self.build_blocks(rb4.shape[1], rb4.shape[2], rb4.shape[3], self.filters[2], 3)(rb4)
        rb6 = self.build_residual_blocks(rb5.shape[1], rb5.shape[2], rb5.shape[3], self.filters[2], 3)(rb5)

        rb7 = self.build_blocks(rb6.shape[1], rb6.shape[2], rb6.shape[3], self.filters[3], 3)(rb6)
        rb8 = self.build_residual_blocks(rb7.shape[1], rb7.shape[2], rb7.shape[3], self.filters[3], 3)(rb7)

        gap1 = GlobalAveragePooling2D()(rb8)
        out = Dense(self.n_output, activation='softmax')(gap1)

        outputs = Model(inputs=inputs, outputs=out)
        return outputs

    def build_residual_blocks(self, n_time, n_freq, n_channel, n_filter, kernel_size):
        inputs = Input(shape=(n_time, n_freq, n_channel,))
        conv1 = Conv2D(n_filter, kernel_size, padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        activation1 = Activation('relu')(bn1)

        conv2 = Conv2D(n_filter, kernel_size, padding='same')(activation1)
        bn2 = BatchNormalization()(conv2)
        added = Add()([bn2, inputs])
        activation2 = Activation('relu')(added)

        outputs = Model(inputs=inputs, outputs=activation2)
        return outputs

    def build_blocks(self, n_time, n_freq, n_channel, n_filter, kernel_size):
        inputs = Input(shape=(n_time, n_freq, n_channel,))
        conv1 = Conv2D(n_filter, kernel_size, strides=2, padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        activation1 = Activation('relu')(bn1)

        conv2 = Conv2D(n_filter, kernel_size, padding='same')(activation1)
        bn2 = BatchNormalization()(conv2)
        activation2 = Activation('relu')(bn2)

        outputs = Model(inputs=inputs, outputs=activation2)
        return outputs

    def call(self, inputs):
        network = self.network(inputs)
        return network


def model_save(param, machine_type, model1, model2, model3, epoch):
    model1_path = '{root}/{model}/epoch_{epoch}/model1_{machine_type}'.format(root=param['model_root'],
                                                                              model=param['model_dir'],
                                                                              epoch=epoch,
                                                                              machine_type=machine_type)
    model2_path = '{root}/{model}/epoch_{epoch}/model2_{machine_type}'.format(root=param['model_root'],
                                                                              model=param['model_dir'],
                                                                              epoch=epoch,
                                                                              machine_type=machine_type)
    model3_path = '{root}/{model}/epoch_{epoch}/model3_{machine_type}'.format(root=param['model_root'],
                                                                              model=param['model_dir'],
                                                                              epoch=epoch,
                                                                              machine_type=machine_type)

    if model1 != 0:
        model1.save(model1_path)
        # print("save_model -> {}".format(model1_path))

    if model2 != 0:
        model2.save(model2_path)
        # print("save_model -> {}".format(model2_path))

    if model3 != 0:
        model3.save(model3_path)
        # print("save_model -> {}".format(model3_path))

    print('Model saved Epoch {}'.format(epoch))


def model1_load(param, machine, epoch):
    path = '{root}/{model}/epoch_{ep}/model1_{machine}'.format(root=param['model_root'],
                                                               model=param['model_dir'],
                                                               ep=epoch,
                                                               machine=machine)
    if not os.path.exists(path):
        print("{} model 1 not found ".format(machine))
        sys.exit(-1)
    model = tf.keras.models.load_model(path)
    return model


def model2_load(param, machine, epoch):
    path = '{root}/{model}/epoch_{ep}/model2_{machine}'.format(root=param['model_root'],
                                                               model=param['model_dir'],
                                                               ep=epoch,
                                                               machine=machine)
    if not os.path.exists(path):
        print("{} model 2 not found ".format(machine))
        sys.exit(-1)
    model = tf.keras.models.load_model(path)
    return model


def model3_load(param, machine, epoch):
    path = '{root}/{model}/epoch_{ep}/model3_{machine}'.format(root=param['model_root'],
                                                               model=param['model_dir'],
                                                               ep=epoch,
                                                               machine=machine)
    if not os.path.exists(path):
        print("{} model 3 not found ".format(machine))
        sys.exit(-1)
    model = tf.keras.models.load_model(path)
    return model