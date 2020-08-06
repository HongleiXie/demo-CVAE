import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from model import CVAE

train_size = 60000
batch_size = 32
test_size = 10000
EPOCHS = 10
latent_dim = 2
inter_dim = 128
# num_examples_to_generate = 16

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# model each pixel with a Bernoulli distribution
def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return images.astype('float32')


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x, y):
    mean, logvar = model.encode(x,y)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z,y)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(model, x_test, y_test):
    test_loss(compute_loss(model, x_test, y_test))

def generate_and_save_images(model, epoch, test_sample_x, test_sample_y):
    mean, logvar = model.encode(test_sample_x, test_sample_y)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z, test_sample_y)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

if __name__ == '__main__':

    (train_images, train_y), (test_images, test_y) = tf.keras.datasets.mnist.load_data()
    train_y = tf.one_hot(train_y, depth=10)
    test_y = tf.one_hot(test_y, depth=10)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices((train_images, train_y))
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_y))
                    .shuffle(test_size).batch(batch_size))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # # test
    # model = CVAE(latent_dim, inter_dim)
    # for test_batch in test_dataset.take(1):
    #     test_sample_x = test_batch[0][0:num_examples_to_generate, :, :, :]
    #     test_sample_y = test_batch[1][0:num_examples_to_generate]

    # generate_and_save_images(model, 0, test_sample_x, test_sample_y)

    # reset
    model = CVAE(latent_dim, inter_dim)

    for epoch in range(EPOCHS):
        for train in train_dataset:
            train_step(model, train[0], train[1], optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

        for test in test_dataset:
            test_step(model, test[0], test[1])
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), test_loss.result()))
        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        # generate_and_save_images(model, epoch, test_sample_x, test_sample_y)

    model.save_weights('saved_model/my_checkpoint')