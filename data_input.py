
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random


def read_labeled_image_batches(FLAGS):
    print("\nReading input images from {0}".format(FLAGS.img_dir))
    print("-----------------------------")
    class DataSet(object):
        pass
    
    # Create dataset structure
    train_data_set = DataSet()
    test_data_set = DataSet()
    data_sets = DataSet()
    data_sets.train = train_data_set
    data_sets.test = test_data_set
    
    # Reads pfathes of images together with their labels
    image_list, label_list, num_classes = _read_labeled_image_list(FLAGS.img_dir)
    image_list, label_list = _shuffle_tow_arrays_together(image_list, label_list)   
    
    train_data_set.num_classes = num_classes
    test_data_set.num_classes = num_classes
    print("Num of classes: {0}".format(num_classes))

    # Split into training and testing sets
    train_images = image_list[FLAGS.test_size:]
    train_labels = label_list[FLAGS.test_size:]
    test_images = image_list[:FLAGS.test_size]
    test_labels = label_list[:FLAGS.test_size]
    assert all(test_image not in train_images for test_image in test_images), "Some images are contained in testing- and training-set." 
    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)
    
    train_data_set.size = len(train_labels)
    test_data_set.size = len(test_labels)
    print ("Num of training images: {0}".format(train_data_set.size))
    print ("Num of testing images: {0}".format(test_data_set.size))
    assert test_data_set.size == FLAGS.test_size, "Number of testing images is too big."
    assert test_data_set.size < train_data_set.size, "More testing images then training images."

    # Read images from disk async (when needed => tensor)
    tf_train_images = ops.convert_to_tensor(train_images, dtype=dtypes.string)
    tf_train_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)
    tf_test_images = ops.convert_to_tensor(test_images, dtype=dtypes.string)
    tf_test_labels = ops.convert_to_tensor(test_labels, dtype=dtypes.int32)

    input_queue_train = tf.train.slice_input_producer([tf_train_images, tf_train_labels], num_epochs=FLAGS.num_epochs)
    input_queue_test = tf.train.slice_input_producer([tf_test_images, tf_test_labels], num_epochs=FLAGS.num_epochs)

    train_images_disk, train_labels_disk = _read_images_from_disk(input_queue_train, FLAGS)
    test_images_disk, test_labels_disk = _read_images_from_disk(input_queue_test, FLAGS)
    
    # Create training batches.
    # Not shuffled because it is already shuffled above. 
    train_image_batch, train_label_batch = tf.train.batch([train_images_disk, train_labels_disk], batch_size=FLAGS.batch_size)
    test_image_batch, test_label_batch = tf.train.batch([test_images_disk, test_labels_disk], batch_size=FLAGS.batch_size)
    
    train_data_set.batch_size = FLAGS.batch_size
    train_data_set.images = train_image_batch
    train_data_set.labels = train_label_batch
    test_data_set.batch_size = FLAGS.batch_size
    test_data_set.images = test_image_batch
    test_data_set.labels = test_label_batch

    print ("Batch size: {0}".format(train_data_set.batch_size))
    print("-----------------------------\n")
    return data_sets                  
   

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
    num_classes = 0
    for label in label_dirs:
        num_classes += 1
        subdir = path + label
        for image in os.listdir(subdir):
            filenames.append("{0}/{1}".format(subdir, image))
            labels.append(int(label))
    
    assert len(filenames) == len(labels), "Supervised training => number of images and labels must be the same"
    return filenames, labels, num_classes


def _shuffle_tow_arrays_together(a, b):
    assert len(a) == len(b), "It is not possible to shuffle two lists with different len together."    

    indexes = list(range(len(a)))
    random.shuffle(indexes)
    
    shuffled_a = []
    shuffled_b = []
    for index in indexes:
        shuffled_a.append(a[index])
        shuffled_b.append(b[index])
    
    return shuffled_a, shuffled_b


def _read_images_from_disk(input_queue, FLAGS):    
    images_queue = input_queue[0]
    labels_queue = input_queue[1]
    files = tf.read_file(images_queue)
    
    if FLAGS.is_jpeg:
        images = tf.image.decode_jpeg(files, channels=3)   
    else:
        images = tf.image.decode_png(files, channels=3)   
        
    images.set_shape([FLAGS.image_height, FLAGS.image_width, 3])
    
    return images, labels_queue 