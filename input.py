
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import resize
import random


def read_labeled_image_batches(FLAGS):
    # Reads pfathes of images together with their labels
    image_list, label_list = _read_labeled_image_list(FLAGS.img_dir)
    image_list, label_list = _shuffle_tow_arrays_together(image_list, label_list)   

    # Split into training and testing sets
    training_images = image_list[:FLAGS.training_size]
    training_labels = label_list[:FLAGS.training_size]
    test_images = image_list[FLAGS.training_size:]
    test_labels = label_list[FLAGS.training_size:]
    
    num_train_data = len(training_images)
    num_test_data = len(test_images)
    print ("Num of training images: {0}".format(num_train_data))
    print ("Num of testing images: {0}".format(num_test_data))

    tf_train_images = ops.convert_to_tensor(training_images, dtype=dtypes.string)
    tf_train_labels = ops.convert_to_tensor(training_labels, dtype=dtypes.int32)
    tf_test_images = ops.convert_to_tensor(test_images, dtype=dtypes.string)
    tf_test_labels = ops.convert_to_tensor(test_labels, dtype=dtypes.int32)

    # Makes an input queue
    input_queue_train = tf.train.slice_input_producer([tf_train_images, tf_train_labels],
                                                shuffle=True)
    input_queue_test = tf.train.slice_input_producer([tf_test_images, tf_test_labels],
                                                shuffle=True)

    train_images_disk, train_labels_disk = _read_images_from_disk(input_queue_train, FLAGS)
    test_images_disk, test_labels_disk = _read_images_from_disk(input_queue_test, FLAGS)

    
    train_images_disk = _process_image(train_images_disk, FLAGS)
    test_images_disk = _process_image(train_images_disk, FLAGS) 

    # Ensure that the random shuffling has good mixing properties.
    num_preprocess_threads = 16
    
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_train_data * min_fraction_of_examples_in_queue) 
    train_image_batch, train_label_batch = tf.train.shuffle_batch(
        [train_images_disk, train_labels_disk], 
        num_threads = num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples,
        batch_size=FLAGS.batch_size)
        
    min_queue_examples = int(num_test_data * min_fraction_of_examples_in_queue)
    test_image_batch, test_label_batch = tf.train.shuffle_batch(
        [test_images_disk, test_labels_disk], 
        num_threads = num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples,
        batch_size=FLAGS.batch_size)
                                        
    return train_image_batch, tf.reshape(train_label_batch, [FLAGS.batch_size]), test_image_batch, tf.reshape(test_label_batch, [FLAGS.batch_size])
   

def _read_labeled_image_list(path):
    """Reads images and labels from file system. Create a folder for each label and put 
       all images with this label into the sub folder (you don't need a label.txt etc.)
       Note: Images can be downloaded with datr - https://github.com/peerdavid/datr
    Args:
      path: Folder, which contains labels (folders) with images.
    Returns:
      List with all filenames and list with all labels
    """
    
    print("Reading all image labels and file names.")
    filenames = []
    labels = []
    label_dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    for label in label_dirs:
        subdir = path + label
        for image in os.listdir(subdir):
            filenames.append("{0}/{1}".format(subdir, image))
            labels.append(int(label))
    
    assert len(filenames) == len(labels)
    return filenames, labels


def _shuffle_tow_arrays_together(a, b):
    indexes = list(range(len(a)))
    random.shuffle(indexes)
    
    shuffled_a = []
    shuffled_b = []
    for index in indexes:
        shuffled_a.append(a[index])
        shuffled_b.append(b[index])
    
    return shuffled_a, shuffled_b


def _read_images_from_disk(input_queue, FLAGS):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    
    print("Reading images from disk.")
    images = input_queue[0]
    labels = input_queue[1]
    file_contents = tf.read_file(images)
    decoded_images = tf.image.decode_jpeg(file_contents, channels=3)   
    decoded_images.set_shape([FLAGS.orig_image_height, FLAGS.orig_image_width, 3])
    
    return decoded_images, labels 

# mogrify -gravity Center -extent 240x150 -background black -colorspace RGB *jpg
def _process_image(image, FLAGS):
    print("Processing images.")
    float_images = tf.cast(image, tf.float32)

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.image_height, FLAGS.image_width)
    float_images = tf.image.resize_images(float_images, FLAGS.image_height, FLAGS.image_width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_images = tf.image.per_image_whitening(float_images)
    return float_images





