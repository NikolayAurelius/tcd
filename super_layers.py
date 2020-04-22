import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Add, BatchNormalization, Activation, Flatten, Dropout, Concatenate, \
    Dense, GaussianNoise, Maximum


def create_shared_weights(conv1, convs, input_shape):
    with tf.name_scope(conv1.name):
        conv1.build(input_shape)

    for conv2 in convs:
        with tf.name_scope(conv2.name):
            conv2.build(input_shape)
        conv2.kernel = conv1.kernel
        conv2.bias = conv1.bias
        # print(conv2._trainable_weights)
        conv2._trainable_weights = []
        conv2._trainable_weights.append(conv2.kernel)
        conv2._trainable_weights.append(conv2.bias)


def create_shared_weights_bn(conv1, convs, input_shape):
    with tf.name_scope(conv1.name):
        conv1.build(input_shape)

    for conv2 in convs:
        with tf.name_scope(conv2.name):
            conv2.build(input_shape)
        conv2.beta = conv1.beta
        conv2.gamma = conv1.gamma
        conv2.moving_mean = conv1.moving_mean
        conv2.moving_variance = conv1.moving_variance
        conv2._trainable_weights = [elem for elem in conv1._trainable_weights]
        # conv2._trainable_weights.append(conv2.kernel)
        # conv2._trainable_weights.append(conv2.bias)


def super_Conv4D(filters, kernel_size, activation='linear', batch_normalization=False, dropout_rate=0.0, noise_rate=0.0,
                 kernel_regularizer=None, bias_regularizer=None):
    a, b, c, d = kernel_size

    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}
        inputs = npts
        if noise_rate > 0.0:
            for key in inputs.keys():
                inputs[key] = [GaussianNoise(noise_rate)(x) for x in inputs[key]]

        if dropout_rate > 0.0:
            for key in inputs.keys():
                inputs[key] = [Dropout(dropout_rate)(x) for x in inputs[key]]

        _l1 = {'x0': [], 'x3': [], 'x6': [], 'x9': [], 'rev_x0': [], 'rev_x3': [], 'rev_x6': [], 'rev_x9': []}

        for i in range(len(inputs['x0'])):
            main_conv = Conv3D(filters, kernel_size=(b, c, d), kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer)
            shared_convs = [Conv3D(filters, kernel_size=(b, c, d), kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer) for _ in range(7)]

            create_shared_weights(main_conv, shared_convs, inputs['x0'][i].shape)

            for key in _l1.keys():
                x = inputs[key][i]
                if key == 'x0':
                    _l1[key].append(main_conv(x))
                else:
                    _l1[key].append(shared_convs.pop()(x))

        for key in _l1.keys():
            _l1[key] = [Add()(_l1[key][i:i + a]) for i in range(len(inputs[key]) - (a - 1))]

        if batch_normalization:
            for i in range(len(_l1['x0'])):

                main_bn = BatchNormalization(axis=-1, scale=True)
                shared_bns = [BatchNormalization(axis=-1, scale=True) for _ in range(7)]

                create_shared_weights_bn(main_bn, shared_bns, _l1['x0'][i].shape)

                for key in _l1.keys():
                    x = _l1[key][i]
                    if key == 'x0':
                        _l1[key][i] = main_bn(x)
                    else:
                        _l1[key][i] = shared_bns.pop()(x)

        return {f'npts_{key}': [Activation(activation)(x) for x in _l1[key]] for key in _l1.keys()}

    return _res


def super_MaxPooling4D(pool_size):
    a, b, c, d = pool_size

    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}

        for key in npts.keys():
            npts[key] = [Maximum()(npts[key][i:i + a]) for i in range(len(inputs[key]) - (a - 1))]
        return {f'npts_{key}': npts[key] for key in npts.keys()}

    return _res


def super_Concatenate():
    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}
        print(npts['x0'])
        return {f'npts_{key}': [Concatenate(axis=-1)(npts[key])] for key in npts.keys()}

    return _res


def super_Flatten():
    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}
        return {f'npts_{key}': [Flatten()(npts[key][0])] for key in npts.keys()}

    return _res


def super_Dropout(dropout_rate):
    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}
        return {f'npts_{key}': [Dropout(dropout_rate)(npts[key][0])] for key in npts.keys()}

    return _res


def super_Dense(**kwargs):
    def _res(npts_x0, npts_x3, npts_x6, npts_x9, npts_rev_x0, npts_rev_x3, npts_rev_x6, npts_rev_x9):
        npts = {'x0': npts_x0, 'x3': npts_x3, 'x6': npts_x6, 'x9': npts_x9, 'rev_x0': npts_rev_x0,
                'rev_x3': npts_rev_x3, 'rev_x6': npts_rev_x6, 'rev_x9': npts_rev_x9}
        l1 = {key: Dense(**kwargs) for key in npts.keys()}

        create_shared_weights(l1['x0'], [l1[key] for key in npts.keys() if key != 'x0'], npts['x0'][0].shape)

        return {f'npts_{key}': [l1[key](npts[key][0])] for key in l1.keys()}

    return _res