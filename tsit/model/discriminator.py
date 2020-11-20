from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

def build_discriminator(input_shape_a=(2048, 1024, 1),
                        input_shape_b=(2048, 1024, 3),
                        input_shape_c=(2048, 1024, 3),
                        ndf=64, n_layers=3,
                        kernel_size=4, strides=2, activation='linear',
                        n_downsampling=1, name='Discriminator'):
    input_a = Input(shape=input_shape_a)
    input_b = Input(shape=input_shape_b)
    input_c = Input(shape=input_shape_c)

    features = []
    x = Concatenate(axis=-1)([input_a, input_b, input_c])
    for _ in range(n_downsampling):
        x = AveragePooling2D(3, strides=2, padding='same')(x)

    x = Conv2D(ndf, kernel_size=kernel_size, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    features.append(x)

    nf = ndf
    for _ in range(1, n_layers):
        nf = min(ndf * 2, 512)
        x = Conv2D(nf, kernel_size=kernel_size, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)

    nf = min(nf * 2, 512)
    x = Conv2D(nf, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    features.append(x)

    x = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = Activation(activation)(x)

    # create model graph
    model = Model(inputs=[input_a, input_b, input_c], outputs=[x] + features, name=name)
    print("\nDiscriminator")
    model.summary()
    return model


if __name__ == "__main__":
    discriminator_0 = build_discriminator(n_downsampling=0)
    discriminator_1 = build_discriminator(n_downsampling=1)
    discriminator_2 = build_discriminator(n_downsampling=2)
    