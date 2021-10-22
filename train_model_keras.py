# Copyright 2020 by Andrey Ignatov. All Rights Reserved.


import tensorflow as tf
import numpy as np
import sys

from load_dataset import image_generator
from model_keras import PyNET
import utils
import losses
from datetime import datetime

# Processing command arguments

LEVEL, batch_size, learning_rate, restore_iter, num_train_iters, \
dataset_dir, dslr_dir, phone_dir, over_dir, under_dir, vgg_dir, \
triple_exposure, vgg_dir, eval_step = \
    utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches

PATCH_WIDTH, PATCH_HEIGHT = 128, 128
PATCH_DEPTH = 4
if triple_exposure:
    PATCH_DEPTH *= 3

DSLR_SCALE = float(1) / (2 ** (LEVEL - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)


# Training and validation data pipelines
train_generator = image_generator(dataset_dir, dslr_dir, phone_dir, 'train/', DSLR_SCALE, PATCH_WIDTH, PATCH_HEIGHT, triple_exposure, over_dir, under_dir)
train_dataset = tf.data.Dataset.from_tensor_slices(train_generator.get_list()).repeat()
train_dataset = train_dataset.shuffle(train_generator.length())
train_dataset = train_dataset.map(train_generator.read,
                                num_parallel_calls=-1)
train_dataset = train_dataset.map(train_generator.augment_image,
                                num_parallel_calls=-1)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=1)

val_generator = image_generator(dataset_dir, dslr_dir, phone_dir, 'val/', DSLR_SCALE, PATCH_WIDTH, PATCH_HEIGHT, triple_exposure, over_dir, under_dir)
val_dataset = tf.data.Dataset.from_tensor_slices(val_generator.get_list())
val_dataset = val_dataset.map(val_generator.read,
                                num_parallel_calls=-1)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(buffer_size=1)

# Defining and compiling the model architecture
gen = PyNET(train_generator.size(), LEVEL)

metrics = [losses.loss_content(vgg_dir), losses.metr_psnr]
if LEVEL < 2:
    metrics.append(losses.loss_ms_ssim)

if LEVEL == 5 or LEVEL == 4:
    loss = losses.loss_creator(PATCH_WIDTH, PATCH_HEIGHT, vgg_dir, mse=100)
if LEVEL == 3 or LEVEL == 2:
    loss = losses.loss_creator(PATCH_WIDTH, PATCH_HEIGHT, vgg_dir, mse=100, content=1)
if LEVEL == 1:
    loss = losses.loss_creator(PATCH_WIDTH, PATCH_HEIGHT, vgg_dir, mse=50, content=1)
if LEVEL == 0:
    loss = losses.loss_creator(PATCH_WIDTH, PATCH_HEIGHT, vgg_dir, mse=100, content=1, ssim=20)

gen.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    loss = loss,
    metrics = metrics
)

time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + time
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
cp_dir = "checkpoints/" + time + '_level_' + str(LEVEL) + 'weights.{epoch:02d}.hdf5'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cp_dir
)

gen.fit(
    x = train_dataset,
    validation_data = val_dataset,
    steps_per_epoch=200,
    epochs = num_train_iters,
    callbacks = [tb_callback, cp_callback]
)