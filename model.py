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

x = {'npts_x0': [], 'npts_x3': [], 'npts_x6': [], 'npts_x9': [], 'npts_rev_x0': [], 'npts_rev_x3': [], 'npts_rev_x6': [], 'npts_rev_x9': []}
for key in x.keys():
  x[key] = [Input((18, 18, 18, 1)) for _ in range(18)]
y_true_npt = Input((2,))


batch_normalization = True
m = 10
kr = 0.005
kr1 = kr / 50
dropout_rate = 0.5

q = super_Conv4D(2, (2, 2, 2, 2), activation='linear', dropout_rate=0.1, noise_rate=0.25, kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**x)

#########
y = super_Conv4D(m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**q)
y = super_Conv4D(m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = y

y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = y

y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = y

y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_constraint=max_norm(1.25, axis=[0, 1, 2, 3]))(**y)
y = y
#########

y = super_Concatenate()(**y)
y = super_Dropout(dropout_rate)(**y)

y = super_Flatten()(**y)

y1 = super_Dense(units=2, activation='softmax')(**y)

y = super_Dense(units=20, activation='tanh', kernel_regularizer=l2(kr1))(**y)
y = super_Dense(units=2, activation='softmax', kernel_regularizer=l2(kr1))(**y)
lst_y = [y[key][0] for key in y.keys()]
lst_y1 = [y1[key][0] for key in y1.keys()]


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


def penalty_equals_error(name=f'penalty_equals_error{np.random.randint(0, 10000)}'):
  y1_npt, y2_npt, y_true = Input(shape=(2,)), Input(shape=(2,)), Input(shape=(2,))

  y1 = standartify()(y1_npt)
  y2 = standartify()(y2_npt)
  y_false = Lambda(lambda x: (x - 1) * (-1))(y_true)

  d1 = Dot(axes=-1)([y1, y_false])
  d2 = Dot(axes=-1)([y2, y_false])

  res = Lambda(lambda x: x[0] * x[1])([d1, d2])
  return Model(inputs=[y1_npt, y2_npt, y_true], outputs=res, name=name)


main0 = Average(name='main0')(lst_y)
main1 = Average(name='main1')(lst_y1)

p = penalty_equals_error()([main0, main1, y_true_npt])

penalty_deviation0 = Lambda(penalty_deviation_func, name='penalty_deviation0')(lst_y)
penalty_deviation1 = Lambda(penalty_deviation_func, name='penalty_deviation1')(lst_y1)

inputs = []
for key in x.keys():
  inputs.extend(x[key])
inputs.append(y_true_npt)

outputs = []
for key in y.keys():
  outputs.extend(y[key])

for key in y1.keys():
  outputs.extend(y1[key])

outputs.append(main0)
outputs.append(main1)
outputs.append(penalty_deviation0)
outputs.append(penalty_deviation1)
outputs.append(p)

model = Model(inputs=inputs, outputs=outputs[::-1])
model.summary()
