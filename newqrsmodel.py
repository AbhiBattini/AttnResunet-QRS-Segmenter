import keras
from keras import Model, Input
from keras.src.layers import Conv1D, Conv1DTranspose, concatenate, MaxPooling1D, Activation, BatchNormalization, \
    Dropout, multiply, add


def attention_gate(inp_1, inp_2, n_intermediate_channels):
    inp_1_conv = Conv1D(n_intermediate_channels, 3, padding='same', kernel_initializer=keras.initializers.HeNormal()) \
        (inp_1)
    inp_2_conv = Conv1D(n_intermediate_channels, 3, padding='same', kernel_initializer=keras.initializers.HeNormal()) \
        (inp_2)
    f = add([inp_1_conv, inp_2_conv])
    f = Activation('relu')(f)
    g = Conv1D(1, 1, padding='same', kernel_initializer=keras.initializers.HeNormal())(f)
    gate = Activation('sigmoid')(g)
    return multiply([inp_2, gate])


def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def residual_block(inputs, num_filters, strides=1):
    x = batchnorm_relu(inputs)
    x = Conv1D(num_filters, 3, padding="same", strides=strides, kernel_initializer=keras.initializers.HeNormal())(x)
    x = batchnorm_relu(x)
    x = Conv1D(num_filters, 3, padding="same", strides=1, kernel_initializer=keras.initializers.HeNormal())(x)
    s = Conv1D(num_filters, 1, padding="same", strides=strides, kernel_initializer=keras.initializers.HeNormal())\
        (inputs)
    x = x + s
    return x


def conv_block(input_tensor, num_filters):
    encoder = Conv1D(num_filters, 3, padding='same', kernel_initializer=keras.initializers.HeNormal())(input_tensor)
    encoder = Activation('relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv1D(num_filters, 3, padding='same', kernel_initializer=keras.initializers.HeNormal())(encoder)
    encoder = Activation('relu')(encoder)
    encoder = BatchNormalization()(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling1D(2, strides=2)(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = Conv1DTranspose(num_filters, 2, strides=2, padding='same', kernel_initializer=keras.initializers.
                              HeNormal())(input_tensor)
    decoder = concatenate([decoder, concat_tensor], axis=-1)
    decoder = conv_block(decoder, num_filters)
    return decoder


def unet(input_shape, num_filters_start=16):
    inputs = Input(shape=input_shape)
    inputs = Dropout(0.2)(inputs)
    encoder_pool0, encoder0 = encoder_block(inputs, num_filters_start)
    encoder_pool1, encoder1 = encoder_block(encoder_pool0, num_filters_start * 4)
    encoder_pool2, encoder2 = encoder_block(encoder_pool1, num_filters_start * 8)
    encoder_pool3, encoder3 = encoder_block(encoder_pool2, num_filters_start * 20)
    center = conv_block(encoder_pool3, num_filters_start * 96)
    # Upsampling and establishing the skip connections
    decoder3 = decoder_block(center, encoder3, num_filters_start * 20)
    attn3 = attention_gate(encoder3, decoder3, num_filters_start * 20)
    res3 = residual_block(attn3, num_filters_start * 20)
    decoder2 = decoder_block(res3, encoder2, num_filters_start * 8)
    attn2 = attention_gate(encoder2, decoder2, num_filters_start * 8)
    res2 = residual_block(attn2, num_filters_start * 8)
    decoder1 = decoder_block(res2, encoder1, num_filters_start * 4)
    attn1 = attention_gate(encoder1, decoder1, num_filters_start * 4)
    res1 = residual_block(attn1, num_filters_start * 4)
    decoder0 = decoder_block(res1, encoder0, num_filters_start)
    # Output
    outputs = Conv1D(1, 1, activation='sigmoid', kernel_initializer=keras.initializers.HeNormal())(decoder0)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
