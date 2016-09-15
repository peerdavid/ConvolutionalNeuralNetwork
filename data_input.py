
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random



class DataSet(object):
    pass
    

def read_image_batches_without_labels_from_file_list(image_list, FLAGS):
    num_images = len(image_list)
    label_list = [0 for i in range(num_images)]
    data_set = _create_batches(image_list, label_list, FLAGS, num_images)
    return data_set


def read_image_batches_with_labels_from_path(path, FLAGS):
    image_list, label_list, num_classes = read_labeled_image_list(path)
    data_set = _create_batches(image_list, label_list, FLAGS)
    data_set.num_classes = num_classes
    return data_set


def read_evaluation_and_train_image_batches(FLAGS):
    print("\nReading input images from {0}".format(FLAGS.img_dir))
    print("-----------------------------")
       
    # Reads pathes of images together with their labels
    image_list, label_list, num_classes = read_labeled_image_list(FLAGS.img_dir)
    image_list, label_list = _shuffle_tow_arrays_together(image_list, label_list)   
    
    # Split into training and ing sets
    train_images = image_list[FLAGS.evaluation_size:]
    train_labels = label_list[FLAGS.evaluation_size:]
    evaluation_images = image_list[:FLAGS.evaluation_size]
    evaluation_labels = label_list[:FLAGS.evaluation_size]

    assert all(evaluation_image not in train_images for evaluation_image in evaluation_images), "Some images are contained in both, evaluation- and training-set." 
    assert len(train_images) == len(train_labels), "Length of train image list and train label list is different"
    assert len(evaluation_images) == len(evaluation_labels), "Length of evaluation image list and train label list is different"

    # Create image and label batches
    train_data_set = _create_batches(train_images, train_labels, FLAGS)
    evaluation_data_set = _create_batches(evaluation_images, evaluation_labels, FLAGS)
    train_data_set.num_classes = num_classes
    evaluation_data_set.num_classes = num_classes

    print("Num of classes: {0}".format(num_classes))
    print("Num of training images: {0}".format(train_data_set.size))
    print("Num of evaluation images: {0}".format(evaluation_data_set.size))
    print("Batch size: {0}".format(train_data_set.batch_size))
    print("-----------------------------\n")

    assert evaluation_data_set.size == FLAGS.evaluation_size, "Number of evaluation images is too big."
    assert evaluation_data_set.size < train_data_set.size, "More evaluation images than training images."

    data_sets = DataSet()
    data_sets.train = train_data_set
    data_sets.evaluation = evaluation_data_set

    return data_sets                  


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


# mogrify -gravity Center -extent 120x75 -background black -colorspace RGB *jpg
def _read_images_from_disk(input_queue, FLAGS):    
    images_queue = input_queue[0]
    labels_queue = input_queue[1]
    files = tf.read_file(images_queue)
    
    if (FLAGS.image_format == 0):
        images = tf.image.decode_jpeg(files, channels=3)   
    else:
        images = tf.image.decode_png(files, channels=3)   
        
    images.set_shape([FLAGS.image_height, FLAGS.image_width, 3])
    
    return images, labels_queue 


def _create_batches(image_list, label_list, FLAGS, batch_size = None):

    if(batch_size is None):
        batch_size = FLAGS.batch_size

    data_set = DataSet()
    data_set.size = len(image_list)
    data_set.batch_size = batch_size
    data_set.image_list = image_list
    data_set.label_list = label_list

    tf_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    tf_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    input_queue = tf.train.slice_input_producer([tf_images, tf_labels], shuffle=False)

    images_disk, lables_disk = _read_images_from_disk(input_queue, FLAGS)

    data_set.images, data_set.labels = tf.train.batch([images_disk, lables_disk], 
            batch_size=data_set.batch_size, num_threads=10)
    return data_set