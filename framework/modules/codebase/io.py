import tensorflow as tf

def random_augmentation(input_data, fliplr = 1, flipud = 1, rot = 1):
    output = tf.image.random_flip_left_right(input_data)
    output = tf.image.random_flip_up_down(output)
    output = tf.image.rot90(output, tf.random.uniform([], minval = 0, maxval = 4, dtype = input_data.dtype))
    return output


def parse_image_from_tfrecord(example_proto):
    features = {
        'raw_byte': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    raw_image = tf.image.decode_image(parsed_features['raw_byte'], channels = 3, dtype=tf.uint8)

    return raw_image

def parse_image_label_from_tfrecord(example_proto):
    features = {
        'raw_byte': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature((), tf.int64, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    raw_image = tf.image.decode_image(parsed_features['raw_byte'], channels = 3, dtype=tf.uint8)
    float_image = tf.cast(raw_image, tf.float32)
    float_image = float_image / 255.0
    label = parsed_features['label']
    
    return float_image, label

def parse_image_view_cluster_label_from_tfrecord(example_proto):
    features = {
        'raw_byte': tf.FixedLenFeature((), tf.string, default_value=''),
        'view_label': tf.FixedLenFeature((), tf.int64, default_value=0),
        'cluster_label': tf.FixedLenFeature((), tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    raw_image = tf.image.decode_image(parsed_features['raw_byte'], channels = None, dtype=tf.uint8)
    float_image = tf.cast(raw_image, tf.float32)
    float_image = float_image / 255.0
    view_label = parsed_features['view_label']
    cluster_label = parsed_features['cluster_label']
    
    return float_image, view_label, cluster_label

def parse_voxel_from_tfrecord(example_proto):
    features = {
        'raw_byte': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    raw_voxel_flattened = tf.image.decode_raw(parsed_features['raw_byte'], out_type=tf.float32)
    
    return raw_voxel_flattened


