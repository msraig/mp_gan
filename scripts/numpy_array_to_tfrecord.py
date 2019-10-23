import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

from framework.utils.io import make_dir
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--numpy_array', type=str, required=True, help='input.')
parser.add_argument('--out_name', type=str, required=True, help='output.')
parser.add_argument('--n_view_bin', type=int, required=True, help='n_view_bin.')
parser.add_argument('--padding', type=int, default=0, help='padding at each dim')

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tf_record(writer, _mask, _view_label, _cluster_label):
    _mask[_mask > 0.5] = 1
    _mask[_mask <= 0.5] = 0
    _, raw_byte = cv2.imencode(".png", _mask * 255)
    raw_byte = raw_byte.tostring()
    
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

    data_array = np.load(args.numpy_array)
    writer = tf.python_io.TFRecordWriter(args.out_name)
    for i in range(data_array.shape[0]):
        _data = data_array[i,...,0]
        _origin_size = _data.shape
        _data = np.pad(_data, args.padding, mode = 'constant')
        # print(_data.shape)
        # print(_origin_size)
        _data = cv2.resize(_data, (_origin_size[1], _origin_size[0]), cv2.INTER_NEAREST)
        _view_bin = np.random.randint(0, args.n_view_bin)
        write_tf_record(writer, _data, _view_bin, 0)




