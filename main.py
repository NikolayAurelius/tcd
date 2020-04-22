from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback, \
    LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from model import model
from generators import base_generator

callbacks = []
batch_size = 32
# TODO: Choice optimizer, start_lr model.train_on_batch

model.compile(optimizer=optimizer(start_lr), loss=None, metrics=None)

model.fit_generator(generator=base_generator(batch_size),
                    steps_per_epoch=100,
                    validation_data=base_generator(batch_size, is_val=True),
                    validation_steps=100,
                    epochs=1000,
                    callbacks=callbacks,
                    verbose=2)
