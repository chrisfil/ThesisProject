from keras.layers import Layer
from keras import backend as K


class PrototypeReconstruction(Layer):
    """
    Reconstruction of the latent space based on the attention maps and
    prototypes (kernel)
    """
    def __init__(self, n_prototypes, **kwargs):
        self.n_prototypes = n_prototypes
        super(PrototypeReconstruction, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        h_shape, alpha_shape = input_shape
        self.kernel = self.add_weight(
            name='prototypes',
            shape=(self.n_prototypes, 1, 1, h_shape[3]),
            initializer='uniform',
            trainable=True
        )
        super(PrototypeReconstruction, self).build(input_shape)

    def call(self, x):
        z, s = x

        s = K.permute_dimensions(s, (3, 0, 1, 2))
        s = K.expand_dims(s, axis=-1)
        kernel = K.expand_dims(self.kernel, axis=1)

        out = K.sum(s * kernel, axis=0)

        z = K.expand_dims(z, axis=1)
        kernel = K.expand_dims(self.kernel, axis=0)

        distance = K.sum(K.pow(z - kernel, 2), axis=(2, 3, 4))

        return [out, distance]

    def get_config(self):
        config = {'n_prototypes': self.n_prototypes}
        base_config = super(PrototypeReconstruction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        h_shape, s_shape = input_shape
        return [h_shape, (h_shape[0], self.n_prototypes)]
