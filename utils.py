
import tensorflow as tf


def create_placeholder_inputs(batch_size, image_height, image_width):
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))
    labels_pl = tf.placeholder(tf.int32, shape=batch_size)
    return images_pl, labels_pl


def create_feed_data(sess, images_placeholder, labels_placeholder, data_set):
    images_r, labels_r = sess.run([data_set.images, data_set.labels])
    return {images_placeholder: images_r, labels_placeholder: labels_r}