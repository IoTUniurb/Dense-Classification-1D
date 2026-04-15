# Imports
import keras

# Blocks
## Base
@keras.utils.register_keras_serializable()
class LinkNetBase1D(keras.Layer):
    def __init__(self, filters: int, **kwargs):
        # Parent constructor
        super().__init__(**kwargs)

        # Configuration
        self.filters = filters

        # Structure
        self.conv = keras.layers.Conv1D(self.filters, 7, strides = 2, padding = 'same')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.mp = keras.layers.MaxPooling1D(2, strides = 2)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })

        return config

## Encoding
@keras.utils.register_keras_serializable()
class LinkNetEncoder1D(keras.Layer):
    def __init__(self, filters: int, kernels: int, **kwargs):
        # Parent constructor
        super().__init__(**kwargs)

        # Configuration
        self.filters = filters
        self.kernels = kernels

        # Structure
        ## Strided skip
        self.skip_conv = keras.layers.Conv1D(self.filters, 1, strides = 2, padding = 'same')

        ## First block
        self.conv_b1_1 = keras.layers.Conv1D(self.filters, self.kernels, strides = 2, padding = 'same')
        self.bn_b1_1 = keras.layers.BatchNormalization()
        self.relu_b1_1 = keras.layers.ReLU()

        self.conv_b1_2 = keras.layers.Conv1D(self.filters, self.kernels, strides = 1, padding = 'same')
        self.bn_b1_2 = keras.layers.BatchNormalization()
        self.relu_b1_2 = keras.layers.ReLU()

        self.add_1 = keras.layers.Add()

        ## Second block
        self.conv_b2_1 = keras.layers.Conv1D(self.filters, self.kernels, strides = 1, padding = 'same')
        self.bn_b2_1 = keras.layers.BatchNormalization()
        self.relu_b2_1 = keras.layers.ReLU()

        self.conv_b2_2 = keras.layers.Conv1D(self.filters, self.kernels, strides = 1, padding = 'same')
        self.bn_b2_2 = keras.layers.BatchNormalization()
        self.relu_b2_2 = keras.layers.ReLU()

        self.add_2 = keras.layers.Add()

    def call(self, inputs):
        skip_1 = self.skip_conv(inputs)

        ## First block
        x = self.conv_b1_1(inputs)
        x = self.bn_b1_1(x)
        x = self.relu_b1_1(x)

        x = self.conv_b1_2(x)
        x = self.bn_b1_2(x)
        x = self.relu_b1_2(x)

        a_1 = self.add_1([skip_1, x])

        ## Second block
        x = self.conv_b2_1(a_1)
        x = self.bn_b2_1(x)
        x = self.relu_b2_1(x)

        x = self.conv_b2_2(x)
        x = self.bn_b2_2(x)
        x = self.relu_b2_2(x)

        out = self.add_2([a_1, x])

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernels': self.kernels,
        })

        return config

## Decoding
@keras.utils.register_keras_serializable()
class LinkNetDecoder1D(keras.Layer):
    def __init__(self, filters: int, kernels: int, matching_filters: int, **kwargs):
        # Parent constructor
        super().__init__(**kwargs)

        # Configuration
        self.filters = filters
        self.kernels = kernels
        self.matching_filters = matching_filters

        # Structure
        self.conv_1 = keras.layers.Conv1D(self.filters, 1, strides = 1, padding='same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()

        self.deconv = keras.layers.Conv1DTranspose(self.filters, self.kernels, strides = 2, padding='same')
        self.bn_2 = keras.layers.BatchNormalization()
        self.relu_2 = keras.layers.ReLU()

        self.conv_2 = keras.layers.Conv1D(self.matching_filters, 1, strides = 1, padding='same')
        self.bn_3 = keras.layers.BatchNormalization()
        self.relu_3 = keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.deconv(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_2(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernels': self.kernels,
            'matching_filters': self.matching_filters,
        })

        return config

# Network
def linknet1d(input_shape: tuple, out_classes: int, filters: int = 16, kernels: int = 3, normalization: keras.layers.Normalization = None):
    inputs = keras.layers.Input(input_shape)
    x = inputs

    if normalization != None:
        x = normalization(x)

    # Base
    x = LinkNetBase1D(filters)(x)

    # Encoders
    e_1 = LinkNetEncoder1D(filters, kernels)(x)
    e_2 = LinkNetEncoder1D(filters * 2, kernels)(e_1)
    e_3 = LinkNetEncoder1D(filters * 4, kernels)(e_2)
    e_4 = LinkNetEncoder1D(filters * 8, kernels)(e_3)

    # Decoders
    d_4 = LinkNetDecoder1D(filters * 2, kernels, filters * 4)(e_4)
    a_3_4 = keras.layers.Add()([e_3, d_4])

    d_3 = LinkNetDecoder1D(filters // 2, kernels, filters * 2)(a_3_4)
    a_2_3 = keras.layers.Add()([e_2, d_3])

    d_2 = LinkNetDecoder1D(filters // 8, kernels, filters)(a_2_3)
    a_1_2 = keras.layers.Add()([e_1, d_2])

    d_1 = LinkNetDecoder1D(filters // 16, kernels, filters // 2)(a_1_2)


    # Classifier
    x = keras.layers.Conv1DTranspose(filters // 2, 3, strides = 2, padding = 'same')(d_1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(filters // 2, 3, strides = 1, padding = 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1DTranspose(out_classes, 2, strides = 2, padding = 'same')(x)
    x = keras.layers.Softmax()(x)

    return keras.Model(inputs = inputs, outputs = x)
