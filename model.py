import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Add, Input, BatchNormalization, Activation, Flatten, Dropout, Concatenate, \
    Dense, GaussianNoise, Reshape, Multiply, Average, Lambda, Dot, RepeatVector, Conv2D, GlobalAveragePooling2D, Maximum
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm, MinMaxNorm, NonNeg
from super_layers import super_Conv4D, super_Concatenate, super_Dense, super_Dropout, super_Flatten, super_MaxPooling4D

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

q = super_Conv4D(2, (2, 2, 2, 2), activation='linear', dropout_rate=0.1, noise_rate=0.25, kernel_regularizer=l2(kr))(**x)

#########
y = super_Conv4D(m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**q)
y = super_Conv4D(m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_regularizer=l2(kr))(**y)
y = y

y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(2 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_regularizer=l2(kr))(**y)
y = y

y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(3 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_regularizer=l2(kr))(**y)
y = y

y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', kernel_regularizer=l2(kr))(**y)
y = super_Conv4D(4 * m, (2, 2, 2, 2), activation='relu', batch_normalization=batch_normalization, kernel_regularizer=l2(kr))(**y)
y = y
#########

g = super_Conv4D(m, (4, 4, 4, 4), activation='relu', kernel_regularizer=l2(kr))(**q)

y = super_Concatenate()(**y)
y = super_Dropout(dropout_rate)(**y)

y = super_Flatten()(**y)

y = super_Dense(units=20, activation='tanh', kernel_regularizer=l2(kr1))(**y)
y = super_Dense(units=2, activation='softmax', kernel_regularizer=l2(kr1))(**y)

main = Average(name='main')([y[key][0] for key in y.keys()])


inputs = []
for key in x.keys():
  inputs.extend(x[key])
inputs.append(y_true_npt)

outputs = []
for key in y.keys():
  outputs.extend(y[key])

outputs.append(main)

model = Model(inputs=inputs, outputs=outputs[::-1])
model.summary()
