
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 3

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 62
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_labeled_image_batches(FLAGS):
    # Reads pfathes of images together with their labels
    image_list, label_list = _read_labeled_image_list(FLAGS.img_dir)

    tf_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    tf_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([tf_images, tf_labels],
                                                shuffle=True)

    image, label = _read_images_from_disk(input_queue)

    float_image = _process_image(image)

    # Ensure that the random shuffling has good mixing properties.
    num_preprocess_threads = 16
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
                            
    image_batch, label_batch = tf.train.shuffle_batch(
        [float_image, label], 
        num_threads = num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples,
        batch_size=FLAGS.batch_size)
                                        
    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])
   

def _read_labeled_image_list(path):
    """Reads images and labels from file system. Create a folder for each label and put 
       all images with this label into the sub folder (you don't need a label.txt etc.)
       Note: Images can be downloaded with datr - https://github.com/peerdavid/datr
    Args:
      path: Folder, which contains labels (folders) with images.
    Returns:
      List with all filenames and list with all labels
    """
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
  


def _read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    decoded = tf.image.decode_jpeg(file_contents, channels=3)
    decoded.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return decoded, label 


def _process_image(image):
    reshaped_image = tf.cast(image, tf.float32)

    # Randomly crop a [height, width] section of the image.
    #distorted_image = tf.random_crop(reshaped_image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, IMAGE_SIZE, IMAGE_SIZE)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    return float_image