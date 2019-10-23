import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

from framework.utils.io import make_dir
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', type=str, required=True, help='folder.')
parser.add_argument('--out_name', type=str, required=True, help='output.')
parser.add_argument('--n_view_bin', type=int, required=True, help='n_view_bin.')
parser.add_argument('--padding', type=int, default=0, help='padding at each dim')

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tf_record(writer, raw_byte, _view_label, _cluster_label):
    
    d_feature = {
        'raw_byte': _bytes_feature(raw_byte),
        'view_label': _int64_feature(_view_label),
        'cluster_label': _int64_feature(_cluster_label)
    }  

    features = tf.train.Features(feature = d_feature)
    example = tf.train.Example(features = features)
    serialized = example.SerializeToString()
    writer.write(serialized)    

if __name__ == '__main__':
    args = parser.parse_args()
    out_folder, out_name = os.path.split(args.out_name)
    make_dir(out_folder)

    filename_list = glob.glob(args_dict.img_folder + r'/*.png') + glob.glob(args_dict.img_folder + r'/*.jpg')
    writer = tf.python_io.TFRecordWriter(args.out_name)

    for filename in filename_list:
        print(filename)
        with open(filename, 'rb') as f:
            raw_byte = f.read()

        _view_bin = np.random.randint(0, args.n_view_bin)
        write_tf_record(writer, _data, _view_bin, 0)