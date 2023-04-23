from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

from tqdm.keras import TqdmCallback

from dataset import get_dataset
from figure import generate_reconstructions, generate_error_chart


image_shape = 84, 84
embedding_size = 50

n_train = 10000
n_test = 2000


# This class was copied from https://keras.io/examples/generative/vae/
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

kl_weight = 0.00005

# We decompose the KL loss into two additive terms, for the mean and log variance.
# This is done to make it easier to use with Keras.
def kl_mean_loss(_, z_mean):
    x = kl_weight * tf.square(z_mean)
    return tf.reduce_mean(x)


def kl_log_var_loss(_, z_log_var):
    x = -kl_weight * (1 + z_log_var - tf.exp(z_log_var))
    return tf.reduce_mean(x)


def make_encoder(variational):
    input = layers.Input(shape=(image_shape[0], image_shape[1], 1))
    x = input

    x = layers.Conv2D(32, (3, 3), activation="elu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(embedding_size, (3, 3), activation="elu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(embedding_size, activation="elu")(x)

    z_mean = layers.Dense(embedding_size, name="z_mean")(x)
    if variational:
        z_log_var = layers.Dense(embedding_size, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(input, [z_mean, z_log_var, z])
    else:
        return Model(input, z_mean)


def make_decoder():
    input = layers.Input(shape=embedding_size)
    x = input

    start_shape = image_shape[0] // 2, image_shape[1] // 2
    x = layers.Dense(start_shape[0] * start_shape[1] * embedding_size, activation="elu")(x)
    x = layers.Reshape((start_shape[0], start_shape[1], embedding_size))(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="elu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=1, activation="elu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    return Model(input, x)


def train_autoencoder(model_path, images, variational=False, use_kl_loss=False):
    encoder = make_encoder(variational)
    decoder = make_decoder()

    input = layers.Input(shape=(image_shape[0], image_shape[1], 1))
    if variational:
        z_mean, z_log_var, embedding = encoder(input)
    else:
        embedding = encoder(input)
    output = decoder(embedding)

    if use_kl_loss:
        loss = ["mean_squared_error", kl_mean_loss, kl_log_var_loss]
        autoencoder = Model(input, [output, z_mean, z_log_var])
    else:
        loss = "mean_squared_error"
        autoencoder = Model(input, output)

    autoencoder.compile(optimizer=Adam(), loss=loss)
    autoencoder.fit(callbacks=[TqdmCallback(verbose=0)],
        x=images,
        y=images,
        epochs=200,
        batch_size=128,
        shuffle=True,
        verbose=0,
    )

    encoder.save(model_path + "/encoder")
    decoder.save(model_path + "/decoder")

    return encoder, decoder


if __name__ == "__main__":
    train_images, test_images = get_dataset(n_train, n_test)

    print("Training standard autoencoder...")
    standard_ae = train_autoencoder("standard", train_images)

    print("Training variational autoencoder...")
    variational_ae = train_autoencoder("variational", train_images, variational=True)

    print("Training VAE with KL loss...")
    kl_loss_ae = train_autoencoder("kl_loss", train_images, variational=True, use_kl_loss=True)

    print("Generating figures...")
    autoencoders = [standard_ae, variational_ae, kl_loss_ae]
    generate_reconstructions(test_images[:6], autoencoders)
    generate_error_chart(test_images, autoencoders)
