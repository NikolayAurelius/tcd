from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback, \
    LearningRateScheduler
from tensorflow.keras.optimizers import Adam, Nadam
from .model import create_super_model
from .generators import base_generator, generator, get_validation
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
batch_size = 32
# TODO: Choice optimizer, start_lr model.train_on_batch

start_lr = 0.1 ** 5
optimizer = Adam

losses = {}
metrics = {}
weights = {}

model, a, b = create_super_model()


for output in model.outputs:
    name, _ = output.name.split('/')
    if 'penalty' in output.name:
        losses[name] = 'mse'
        weights[name] = 1.0
    else:
        losses[name] = 'categorical_crossentropy'
        metrics[name] = 'categorical_accuracy'
        weights[name] = 200.0

        if 'main' in output.name and 'sub' not in output.name:
            weights[name] = 1000.0
        elif 'sub' in output.name:
            weights[name] = 500.0

# model.load_weights('/content/drive/My Drive/renova/real_dataset/models/8170_epoch-38_loss-117.8359_main_c_a-0.7783.ckpt')
model.compile(optimizer=optimizer(start_lr), loss=losses, metrics=metrics, loss_weights=weights)

model.fit(generator(batch_size, a, b),
          epochs=3000,
          verbose=1,
          callbacks=callbacks,
          validation_data=get_validation(a, b),
          steps_per_epoch=2048 // batch_size)
