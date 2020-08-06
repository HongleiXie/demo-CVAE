import numpy as np
import tensorflow as tf

class CVAE(tf.keras.Model):

    def __init__(self, latent_dim, inter_dim):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.inter_dim = inter_dim

        input_y = tf.keras.layers.Input(shape=(10,))
        img = tf.keras.layers.Input((28, 28, 1))

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(img)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        concat_layer = tf.keras.layers.concatenate([x, input_y], axis=1)
        x = tf.keras.layers.Dense(self.inter_dim, activation='relu')(concat_layer)
        encoded = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
        self.encoder = tf.keras.Model(inputs=[img, input_y], outputs=[encoded])  # mu and log_var

        embedd = tf.keras.layers.Input((self.latent_dim,))
        merged_input = tf.keras.layers.Concatenate()([embedd, input_y])

        x = tf.keras.layers.Dense(self.inter_dim)(merged_input)
        x = tf.keras.layers.Dense(7 * 7 * 32, activation='relu')(x)
        x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu')(x)

        x = tf.keras.layers.Conv2DTranspose(filters=32,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same')(x)

        self.decoder = tf.keras.Model(inputs=[embedd, input_y], outputs=[x])

    # Sample z ~ Q(z|X,y)
    @tf.function
    def sample(self, eps, y):
        return self.decode(eps, y, apply_sigmoid=True)

    def encode(self, x, y):
        mean, logvar = tf.split(self.encoder([x, y]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, y, apply_sigmoid=False):
        logits = self.decoder([z, y])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
