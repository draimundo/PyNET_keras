import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def PyNET(input_shape, LEVEL, instance_norm=True, instance_norm_level_1=False):
    input = keras.Input(shape=input_shape)
    # -----------------------------------------
    # Downsampling layers
    conv_l1_d1 = _conv_multi_block(input, 3, num_maps=32, instance_norm=False)              # 224 -> 224
    pool1 = _max_pool(conv_l1_d1, 2)                                                         # 224 -> 112

    conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=64, instance_norm=instance_norm)      # 112 -> 112
    pool2 = _max_pool(conv_l2_d1, 2)                                                         # 112 -> 56

    conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=128, instance_norm=instance_norm)     # 56 -> 56
    pool3 = _max_pool(conv_l3_d1, 2)                                                         # 56 -> 28

    conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=256, instance_norm=instance_norm)     # 28 -> 28
    pool4 = _max_pool(conv_l4_d1, 2)                                                         # 28 -> 14

    # -----------------------------------------
    # Processing: Level 5,  Input size: 14 x 14
    conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=512, instance_norm=instance_norm)
    conv_l5_d2 = layers.add([_conv_multi_block(conv_l5_d1, 3, num_maps=512, instance_norm=instance_norm), conv_l5_d1])
    conv_l5_d3 = layers.add([_conv_multi_block(conv_l5_d2, 3, num_maps=512, instance_norm=instance_norm), conv_l5_d2])
    conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=512, instance_norm=instance_norm)

    conv_t4a = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)      # 14 -> 28
    conv_t4b = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)      # 14 -> 28

    # -> Output: Level 5
    conv_l5_out = _conv_layer(conv_l5_d4, 3, 3, 1, relu=False, instance_norm=False)
    output_l5 = tf.keras.activations.tanh(conv_l5_out) * 0.58 + 0.5

    # # -----------------------------------------
    # # Processing: Level 4,  Input size: 28 x 28
    conv_l4_d2 = layers.concatenate([conv_l4_d1, conv_t4a])
    conv_l4_d3 = _conv_multi_block(conv_l4_d2, 3, num_maps=256, instance_norm=instance_norm)
    conv_l4_d4 = layers.add([_conv_multi_block(conv_l4_d3, 3, num_maps=256, instance_norm=instance_norm), conv_l4_d3])
    conv_l4_d5 = layers.add([_conv_multi_block(conv_l4_d4, 3, num_maps=256, instance_norm=instance_norm), conv_l4_d4])
    conv_l4_d6 = layers.concatenate([_conv_multi_block(conv_l4_d5, 3, num_maps=256, instance_norm=instance_norm), conv_t4b])

    conv_l4_d7 = _conv_multi_block(conv_l4_d6, 3, num_maps=256, instance_norm=instance_norm)

    conv_t3a = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)      # 28 -> 56
    conv_t3b = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)      # 28 -> 56

    # -> Output: Level 4
    conv_l4_out = _conv_layer(conv_l4_d7, 3, 3, 1, relu=False, instance_norm=False)
    output_l4 = tf.keras.activations.tanh(conv_l4_out) * 0.58 + 0.5

    # -----------------------------------------
    # Processing: Level 3,  Input size: 56 x 56
    conv_l3_d2 = layers.concatenate([conv_l3_d1, conv_t3a])
    conv_l3_d3 = layers.add([_conv_multi_block(conv_l3_d2, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d2])
    conv_l3_d4 = layers.add([_conv_multi_block(conv_l3_d3, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d3])
    conv_l3_d5 = layers.add([_conv_multi_block(conv_l3_d4, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d4])
    conv_l3_d6 = layers.concatenate([_conv_multi_block(conv_l3_d5, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d1])
    conv_l3_d7 = layers.concatenate([conv_l3_d6, conv_t3b])

    conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=128, instance_norm=instance_norm)

    conv_t2a = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)       # 56 -> 112
    conv_t2b = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)       # 56 -> 112

    # -> Output: Level 3
    conv_l3_out = _conv_layer(conv_l3_d8, 3, 3, 1, relu=False, instance_norm=False)
    output_l3 = tf.keras.activations.tanh(conv_l3_out) * 0.58 + 0.5

    # -------------------------------------------
    # Processing: Level 2,  Input size: 112 x 112
    conv_l2_d2 = layers.concatenate([conv_l2_d1, conv_t2a])
    conv_l2_d3 = layers.concatenate([_conv_multi_block(conv_l2_d2, 5, num_maps=64, instance_norm=instance_norm), conv_l2_d1])

    conv_l2_d4 = layers.add([_conv_multi_block(conv_l2_d3, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d3])
    conv_l2_d5 = layers.add([_conv_multi_block(conv_l2_d4, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d4])
    conv_l2_d6 = layers.add([_conv_multi_block(conv_l2_d5, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d5])
    conv_l2_d7 = layers.concatenate([_conv_multi_block(conv_l2_d6, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d1])

    conv_l2_d8 = layers.concatenate([_conv_multi_block(conv_l2_d7, 5, num_maps=64, instance_norm=instance_norm), conv_t2b])
    conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=64, instance_norm=instance_norm)

    conv_t1a = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)       # 112 -> 224
    conv_t1b = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)       # 112 -> 224

    # -> Output: Level 2
    conv_l2_out = _conv_layer(conv_l2_d9, 3, 3, 1, relu=False, instance_norm=False)
    output_l2 = tf.keras.activations.tanh(conv_l2_out) * 0.58 + 0.5

    # -------------------------------------------
    # Processing: Level 1,  Input size: 224 x 224
    conv_l1_d2 = layers.concatenate([conv_l1_d1, conv_t1a])
    conv_l1_d3 = layers.concatenate([_conv_multi_block(conv_l1_d2, 5, num_maps=32, instance_norm=False), conv_l1_d1])

    conv_l1_d4 = _conv_multi_block(conv_l1_d3, 7, num_maps=32, instance_norm=False)

    conv_l1_d5 = _conv_multi_block(conv_l1_d4, 9, num_maps=32, instance_norm=instance_norm_level_1)
    conv_l1_d6 = layers.add([_conv_multi_block(conv_l1_d5, 9, num_maps=32, instance_norm=instance_norm_level_1), conv_l1_d5])
    conv_l1_d7 = layers.add([_conv_multi_block(conv_l1_d6, 9, num_maps=32, instance_norm=instance_norm_level_1), conv_l1_d6])
    conv_l1_d8 = layers.add([_conv_multi_block(conv_l1_d7, 9, num_maps=32, instance_norm=instance_norm_level_1), conv_l1_d7])

    conv_l1_d9 = layers.concatenate([_conv_multi_block(conv_l1_d8, 7, num_maps=32, instance_norm=False), conv_l1_d1])

    conv_l1_d10 = layers.concatenate([_conv_multi_block(conv_l1_d9, 5, num_maps=32, instance_norm=False), conv_t1b])
    conv_l1_d11 = layers.concatenate([conv_l1_d10, conv_l1_d1])

    conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=32, instance_norm=False)

    # -> Output: Level 1
    conv_l1_out = _conv_layer(conv_l1_d12, 3, 3, 1, relu=False, instance_norm=False)
    output_l1 = tf.keras.activations.tanh(conv_l1_out) * 0.58 + 0.5

    # ----------------------------------------------------------
    # Processing: Level 0 (x2 upscaling),  Input size: 224 x 224
    conv_l0 = _conv_tranpose_layer(conv_l1_d12, 8, 3, 2)        # 224 -> 448
    conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)

    output_l0 = tf.keras.activations.tanh(conv_l0_out) * 0.58 + 0.5

    if LEVEL == 5:
        output = output_l5
    if LEVEL == 4:
        output = output_l4
    if LEVEL == 3:
        output = output_l3
    if LEVEL == 2:
        output = output_l2
    if LEVEL == 1:
        output = output_l1
    if LEVEL == 0:
        output = output_l0

    return Model(inputs=[input], outputs=[output])

def _conv_tranpose_layer(input, num_filters, filter_size, strides):
    return layers.Conv2DTranspose(num_filters, filter_size, strides, padding="same", activation=tf.nn.leaky_relu, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(input)

def _conv_multi_block(input, max_size, num_maps, instance_norm):
    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:
        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, instance_norm=instance_norm)

        output_tensor = layers.concatenate([output_tensor, conv_5b])

    if max_size >= 7:
        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, instance_norm=instance_norm)

        output_tensor = layers.concatenate([output_tensor, conv_7b])

    if max_size >= 9:
        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, instance_norm=instance_norm)

        output_tensor = layers.concatenate([output_tensor, conv_9b])

    return output_tensor

def _conv_layer(input, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):
    activation = None
    if relu:
        activation = tf.nn.leaky_relu
    x = layers.Conv2D(num_filters, filter_size, strides, padding, activation=activation, \
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01, seed=1), \
        bias_initializer=tf.keras.initializers.constant(0.01))(input)
    if instance_norm:
        x = layers.BatchNormalization()(x)
    return x


def _max_pool(x, n):
    return layers.MaxPool2D(pool_size=n, strides=n, padding='VALID')(x)

if __name__ == "__main__":
    model = PyNET((256,256,4), LEVEL=5)
    dot_img_file = 'tmp/model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, rankdir='LR')