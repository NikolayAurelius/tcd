from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback, \
    LearningRateScheduler
from tensorflow.keras.optimizers import Adam, Nadam
from .model import model
from .generators import base_generator, generator
import os
from time import sleep
from google.colab import drive
import numpy as np

while True:
    try:
        drive.mount('/content/drive', force_remount=True)
        break
    except:
        sleep(1)

base_path = f'/content/drive/My Drive/renova'
models_path = f'{base_path}/real_dataset/models'

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=0.35, min_lr=0.1 ** 32)

saving = ModelCheckpoint(models_path + '/' + str(len(os.listdir(models_path))) + '0_epoch-{epoch:02d}_loss-{loss:.4f}_main_c_a-{main_categorical_accuracy:.4f}.ckpt',
                         monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)


def sch_f(epoch, rate):
    if rate < 1e-8:
        return 0.00001 * 5
    return rate * 0.9725


schedule = LearningRateScheduler(sch_f)

callbacks = [learning_rate_reduction, saving, schedule]
batch_size = 16
# TODO: Choice optimizer, start_lr model.train_on_batch

start_lr = 0.1 ** 4
optimizer = Adam

losses = {}
metrics = {}
weights = {}

for output in model.outputs:
    name, _ = output.name.split('/')
    if 'penalty' in output.name:
        losses[name] = 'mse'
        weights[name] = 0.0001
    else:
        losses[name] = 'categorical_crossentropy'
        metrics[name] = 'categorical_accuracy'
        weights[name] = 1.0

        if 'main' in output.name:
            weights[name] = 200.0

model.compile(optimizer=optimizer(start_lr), loss=losses, metrics=metrics, loss_weights=weights)


import h5py
with h5py.File('tcd/val_set.h5') as f:
    val_X = f['val_X'][:]
    val_Y = f['val_Y'][:]
# val_X = np.expand_dims(val_X, axis=5)

val_X = val_X / np.amax(val_X)
val_X = val_X - np.mean(val_X)

val_X = [val_X[:, i, :, :, :, :] for i in range(18 * 8)]
val_X.append(val_Y)

s = [np.zeros((val_Y.shape[0], 2), dtype=np.float32) for _ in range(3)]
for _ in range(18):
    s.append(val_Y)
val_Y = s


model.fit(generator(batch_size),
          epochs=3000,
          verbose=1,
          callbacks=callbacks,
          validation_data=None,
          steps_per_epoch=1024 // batch_size)
