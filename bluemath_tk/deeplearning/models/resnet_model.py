import keras
from keras import layers
from typing import List, Tuple


def ResidualBlock(width: int) -> layers.Layer:
    def apply(x: layers.Layer) -> layers.Layer:
        input_width = x.shape[3]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)

        x = layers.LayerNormalization(axis=-1, center=True, scale=True)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width: int, block_depth: int) -> layers.Layer:
    def apply(
        x: Tuple[layers.Layer, List[layers.Layer]],
    ) -> Tuple[layers.Layer, List[layers.Layer]]:
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x, skips

    return apply


def UpBlock(width: int, block_depth: int) -> layers.Layer:
    def apply(x: Tuple[layers.Layer, List[layers.Layer]]) -> layers.Layer:
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_model(
    image_height: int,
    image_width: int,
    input_frames: int,
    output_frames: int,
    down_widths: List[int] = [64, 128, 256],
    up_widths: List[int] = [256, 128, 64],
    block_depth: int = 2,
) -> keras.Model:
    """Builds the U-Net like model with residual blocks and skip connections."""

    inputs = keras.Input(shape=(image_height, image_width, input_frames))
    x = layers.Conv2D(down_widths[0], kernel_size=1)(inputs)

    skips = []
    for width in down_widths[:-1]:
        x, skips = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(down_widths[-1])(x)

    for width in up_widths[1:]:
        x = UpBlock(width, block_depth)([x, skips])

    outputs = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(inputs, outputs, name="residual_unet")
