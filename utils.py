
import tensorflow as tf

def create_feed_data(sess, images_placeholder, labels_placeholder, data_set):
    images_r, labels_r = sess.run([data_set.images, data_set.labels])
    return {images_placeholder: images_r, labels_placeholder: labels_r}