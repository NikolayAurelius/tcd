import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Add, Input, BatchNormalization, Activation, Flatten, Dropout, Concatenate, \
    Dense, GaussianNoise, Reshape, Multiply, Average, Lambda, Dot, RepeatVector, Conv2D, GlobalAveragePooling2D, Maximum
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm, MinMaxNorm, NonNeg
from .super_layers import super_Conv4D, super_Concatenate, super_Dense, super_Dropout, super_Flatten, super_MaxPooling4D
from tensorflow.keras.activations import relu
import numpy as np

tf.keras.backend.clear_session()

# batch_normalization = True
# m = 10
kr = 0.1 ** 7
# kr1 = kr / 50
dropout_rate = 0.5



def init_conv_inputs():
    x = {'npts_x0': [], 'npts_x3': [], 'npts_x6': [], 'npts_x9': [], 'npts_rev_x0': [], 'npts_rev_x3': [],
         'npts_rev_x6': [], 'npts_rev_x9': []}
    for key in x.keys():
        x[key] = [Input((18, 18, 18, 1)) for _ in range(18)]
    return x


def number_generator():
    i = 0
    while True:
        yield i
        i += 1


def init_dense_inputs(input_size):
    x = {'npts_x0': [], 'npts_x3': [], 'npts_x6': [], 'npts_x9': [], 'npts_rev_x0': [], 'npts_rev_x3': [],
         'npts_rev_x6': [], 'npts_rev_x9': []}
    for key in x.keys():
        x[key] = [Input((input_size, ))]
    return x


def power_model(m=9, activation='relu', number_gen=number_generator()):
    x = init_conv_inputs()
    y = super_Conv4D(m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**x)
    y = super_Conv4D(m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y)
    y = y

    y = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y)
    y = y

    y = super_Conv4D(3 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(3 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(3 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(3 * m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y)
    y = y

    y = super_Conv4D(4 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(4 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(4 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(4 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y)
    y = super_Conv4D(4 * m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y)
    y = y

    y = super_Concatenate()(**y)
    y = super_Dropout(dropout_rate)(**y)

    y = super_Flatten()(**y)
    return Model(inputs=[x[key] for key in x.keys()],
                 outputs=[y[key] for key in y.keys()],
                 name=f'power_model_{next(number_gen)}')


def third_model(m=10, activation='relu', number_gen=number_generator()):
    x = init_conv_inputs()
    y2 = super_Conv4D(m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**x)
    y2 = super_Conv4D(m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y2)
    y2 = super_Conv4D(m, (8, 8, 8, 8))(**y2)

    y2 = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y2)
    y2 = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, kernel_regularizer=l2(kr))(**y2)
    y2 = super_Conv4D(2 * m, (2, 2, 2, 2), activation=activation, batch_normalization=True, kernel_regularizer=l2(kr))(**y2)
    y2 = super_Conv4D(2 * m, (3, 3, 3, 3), kernel_regularizer=l2(kr))(**y2)

    y2 = super_Concatenate()(**y2)
    y2 = super_Dropout(dropout_rate)(**y2)

    y2 = super_Flatten()(**y2)
    return Model(inputs=[x[key] for key in x.keys()],
                 outputs=[y2[key] for key in y2.keys()],
                 name=f'third_model_{next(number_gen)}')


def lenet_model(number_gen=number_generator()):
    x = init_conv_inputs()
    y = super_Conv4D(6, (3, 3, 3, 3), activation='tanh', kernel_regularizer=l2(kr))(**x)
    y = super_MaxPooling4D((2, 2, 2, 2))(**y)

    y = super_Conv4D(16, (3, 3, 3, 3), activation='tanh', kernel_regularizer=l2(kr))(**y)
    y = super_MaxPooling4D((2, 2, 2, 2))(**y)

    y = super_Conv4D(90, (2, 2, 2, 2), activation='tanh', kernel_regularizer=l2(kr))(**y)

    y = super_Concatenate()(**y)
    y = super_Dropout(dropout_rate)(**y)

    y = super_Flatten()(**y)
    return Model(inputs=[x[key] for key in x.keys()],
                 outputs=[y[key] for key in y.keys()],
                 name=f'lenet_model_{next(number_gen)}')


def full_lenet_model(number_gen=number_generator()):
    x = init_conv_inputs()
    model = lenet_model()
    dd = duble_dense(model.outputs[0].shape[-1], units=40)

    return Model(inputs=x, outputs=dd(model(x)),
                 name=f'full_lenet_model_{next(number_gen)}')


def duble_dense(input_size, units=20, number_gen=number_generator()):
    x = init_dense_inputs(input_size)
    y = super_Dense(units=units, activation='tanh')(**x)
    y = super_Dense(units=2, activation='softmax')(**y)
    return Model(inputs=[x[key] for key in x.keys()],
                 outputs=[y[key] for key in y.keys()],
                 name=f'duble_dense_{next(number_gen)}')


def duble_triple_dense(input_size, units0=20, units1=10, number_gen=number_generator()):
    x = init_dense_inputs(input_size)
    y = super_Dense(units=units0, activation='tanh')(**x)
    y1 = super_Dense(units=units1, activation='tanh')(**y)

    for key in y1.keys():
        y1[key].extend(y[key])

    y = super_Concatenate()(**y1)
    y = super_Dense(units=2, activation='softmax')(**y)
    return Model(inputs=[x[key] for key in x.keys()],
                 outputs=[y[key] for key in y.keys()],
                 name=f'duble_triple_dense_{next(number_gen)}')



# lst_y = [y[key][0] for key in y.keys()]
# lst_y1 = [y1[key][0] for key in y1.keys()]


def penalty_deviation_func(x):
    s = []
    for i in range(8):
        for j in range(8):
            if i + j > 8:
                continue
            s.append(relu(tf.linalg.norm(x[i] - x[j], axis=1, keepdims=True), max_value=1000.0))
    return sum(s)


def standartify():
    standartizator = Sequential()
    standartizator.add(Input(shape=(2,)))
    standartizator.add(Lambda(lambda x: relu(x - 0.4999) * 100))
    return standartizator


def penalty_equals_error(weight=0.01, number_gen=number_generator()):
    y1_npt, y2_npt, y_true = Input(shape=(2,)), Input(shape=(2,)), Input(shape=(2,))

    y1 = standartify()(y1_npt)
    y2 = standartify()(y2_npt)
    y_false = Lambda(lambda x: (x - 1) * (-1))(y_true)

    d1 = Dot(axes=-1)([y1, y_false])
    d2 = Dot(axes=-1)([y2, y_false])

    res = Lambda(lambda x: x[0] * x[1] * weight)([d1, d2])
    return Model(inputs=[y1_npt, y2_npt, y_true], outputs=res, name=f'penalty_equals_error_{next(number_gen) + 1}')


def create_super_model():
    x = init_conv_inputs()
    y_true_npt = Input((2,))

    m1 = power_model(9, 'relu')
    m1 = duble_dense(input_size=m1.outputs[0].shape[-1], units=20)(m1(x))
    m1 = [elem[0] for elem in m1]
    main1 = Average(name='submain1')(m1)
    penalty_deviation1 = Lambda(penalty_deviation_func, name='penalty_deviation1')(m1)

    m2 = power_model(8, 'relu')
    m2 = duble_dense(input_size=m2.outputs[0].shape[-1], units=20)(m2(x))
    m2 = [elem[0] for elem in m2]
    main2 = Average(name='submain2')(m2)
    penalty_deviation2 = Lambda(penalty_deviation_func, name='penalty_deviation2')(m2)

    m3 = power_model(9, 'relu')
    m3 = duble_triple_dense(input_size=m3.outputs[0].shape[-1], units0=20, units1=10)(m3(x))
    m3 = [elem[0] for elem in m3]
    main3 = Average(name='submain3')(m3)
    penalty_deviation3 = Lambda(penalty_deviation_func, name='penalty_deviation3')(m3)

    p1 = penalty_equals_error()([main1, main2, y_true_npt])
    p2 = penalty_equals_error()([main1, main3, y_true_npt])
    p3 = penalty_equals_error()([main2, main3, y_true_npt])

    q1 = full_lenet_model()(x)
    q1 = [elem[0] for elem in q1]
    main4 = Average(name='submain4')(q1)
    penalty_deviation4 = Lambda(penalty_deviation_func, name='penalty_deviation4')(q1)

    q2 = lenet_model()
    q2 = duble_triple_dense(q2.outputs[0].shape[-1], units0=20, units1=10)(q2(x))
    q2 = [elem[0] for elem in q2]
    main5 = Average(name='submain5')(q2)
    penalty_deviation5 = Lambda(penalty_deviation_func, name='penalty_deviation5')(q2)

    q3 = lenet_model()
    q3 = duble_triple_dense(q3.outputs[0].shape[-1], units0=30, units1=5)(q3(x))
    q3 = [elem[0] for elem in q3]
    main6 = Average(name='submain6')(q3)
    penalty_deviation6 = Lambda(penalty_deviation_func, name='penalty_deviation6')(q3)

    p4 = penalty_equals_error()([main4, main5, y_true_npt])
    p5 = penalty_equals_error()([main4, main6, y_true_npt])
    p6 = penalty_equals_error()([main5, main6, y_true_npt])

    g1 = third_model(9, 'relu')
    g1 = duble_dense(input_size=g1.outputs[0].shape[-1], units=20)(g1(x))
    g1 = [elem[0] for elem in g1]
    main7 = Average(name='submain7')(g1)
    penalty_deviation7 = Lambda(penalty_deviation_func, name='penalty_deviation7')(g1)

    g2 = third_model(8, 'relu')
    g2 = duble_dense(input_size=g2.outputs[0].shape[-1], units=20)(g2(x))
    g2 = [elem[0] for elem in g2]
    main8 = Average(name='submain8')(g2)
    penalty_deviation8 = Lambda(penalty_deviation_func, name='penalty_deviation8')(g2)

    g3 = third_model(9, 'relu')
    g3 = duble_triple_dense(input_size=g3.outputs[0].shape[-1], units0=20, units1=10)(g3(x))
    g3 = [elem[0] for elem in g3]
    main9 = Average(name='submain9')(g3)
    penalty_deviation9 = Lambda(penalty_deviation_func, name='penalty_deviation9')(g3)

    p7 = penalty_equals_error()([main7, main8, y_true_npt])
    p8 = penalty_equals_error()([main7, main9, y_true_npt])
    p9 = penalty_equals_error()([main8, main9, y_true_npt])

    true_main1 = Average(name='main1')([main1, main2, main3])
    true_main2 = Average(name='main2')([main4, main5, main6])
    true_main3 = Average(name='main3')([main7, main8, main9])

    true_main = Average(name='main')([true_main1, true_main2, true_main3])

    pairs_lst = []

    for elem in [main1, main2, main3]:
        for jelem in [main4, main5, main6]:
            pairs_lst.append([elem, jelem])

    for elem in [main1, main2, main3]:
        for jelem in [main7, main8, main9]:
            pairs_lst.append([elem, jelem])

    for elem in [main4, main5, main6]:
        for jelem in [main7, main8, main9]:
            pairs_lst.append([elem, jelem])

    p_lst = []
    for pairs in pairs_lst:
        p_lst.append(penalty_equals_error(1.0)([pairs[0], pairs[1], y_true_npt]))

    inputs = []
    for key in x.keys():
        inputs.append(x[key])
    inputs.append(y_true_npt)

    outputs = []

    outputs.append(penalty_deviation1)
    outputs.append(penalty_deviation2)
    outputs.append(penalty_deviation3)
    outputs.append(p1)
    outputs.append(p2)
    outputs.append(p3)

    outputs.append(penalty_deviation4)
    outputs.append(penalty_deviation5)
    outputs.append(penalty_deviation6)
    outputs.append(p4)
    outputs.append(p5)
    outputs.append(p6)

    outputs.append(penalty_deviation7)
    outputs.append(penalty_deviation8)
    outputs.append(penalty_deviation9)
    outputs.append(p7)
    outputs.append(p8)
    outputs.append(p9)

    outputs.extend(p_lst)

    a = len(outputs)

    outputs.extend(m1)
    outputs.extend(m2)
    outputs.extend(m3)
    outputs.extend(q1)
    outputs.extend(q2)
    outputs.extend(q3)

    outputs.append(main1)
    outputs.append(main2)
    outputs.append(main3)
    outputs.append(main4)
    outputs.append(main5)
    outputs.append(main6)

    outputs.append(true_main1)
    outputs.append(true_main2)
    outputs.append(true_main3)

    outputs.append(true_main)

    b = len(outputs) - a
    return Model(inputs=inputs, outputs=outputs, name='super_model'), a, b  # Numbers of zero and classify outputs
