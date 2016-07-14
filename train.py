
#
# Classify cars with tensorflow
#
# 0 = oldtimer
# 1 = super
# 2 = estate



import os
import traceback
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import random_ops


NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_labeled_image_list(path):
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
  


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label 




#
# M A I N
#
if __name__ == '__main__':
    try:
        # Reads pfathes of images together with their labels
        image_list, label_list = read_labeled_image_list("data/")

        images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
        labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    shuffle=True)

        image, label = read_images_from_disk(input_queue)

        # Randomly crop a [height, width] section of the image.
        height = 100 
        width = 120
        distorted_image = tf.random_crop(image, [height, width, 3])
        
        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(distorted_image, width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_whitening(resized_image)

        # Ensure that the random shuffling has good mixing properties.
        batch_size = 128
        num_preprocess_threads = 16
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                min_fraction_of_examples_in_queue)
                                
        image_batch, label_batch = tf.train.shuffle_batch(
            [float_image, label], 
            num_threads = num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            batch_size=batch_size)
                                               
        # Display the training images in the visualizer.
        tf.image_summary('images', image_batch)
        
        ret1 = image_batch
        ret2 = tf.reshape(label_batch, [batch_size])
         
         
        #with tf.Session() as sess:
        #    result = sess.run(input_queue)
            
    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")

