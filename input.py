
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import resize


def read_labeled_image_batches(FLAGS):
    # Reads pfathes of images together with their labels
    image_list, label_list = _read_labeled_image_list(FLAGS.img_dir)

    tf_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    tf_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([tf_images, tf_labels],
                                                shuffle=True)

    image, label = _read_images_from_disk(input_queue, FLAGS)

    image = _process_image(image, FLAGS)

    # Ensure that the random shuffling has good mixing properties.
    num_preprocess_threads = 16
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
                  
    print ('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)
                   
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], 
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





