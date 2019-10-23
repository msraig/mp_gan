import os, sys

root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), r'../', r'../', r'../', r'../'))
print(root_path)
sys.path.append(root_path)

import json, math
import tensorflow as tf
from framework.utils.io import save_pfm, write_vox, load_vox, make_dir
from framework.modules.codebase.render.voxel_renderer import render_silouette_transparent as render_silouette#, render_silouette_transparent #render_silouette_drc
import tensorflow_graphics as tfg
import numpy as np
import cv2
import pylab as plt


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.gather_nd(a, index_a)))
# exit()

test_voxel = load_vox(os.path.join(root_path, 'test.vox'))#np.load(os.path.join(root_path, r'resources', r'generated_data.npy'))
test_voxel = test_voxel[...,np.newaxis]
batchsize = 6
voxel = np.stack([test_voxel] * batchsize)
print(voxel.shape)
image_size = [64, 64]
test_mod = 'orth'
# test_mod = 'perspective'

# test object transform
# dumproot = os.path.join(root_path, r'modules', r'projection', r'object_transform_test')
# camera_azimuth_list =  [0.0] * batchsize
# camera_elevation_list = [0.0] * batchsize
# object_azimuth_list = [0.0,   0.0,            math.pi/4,   math.pi/4,      math.pi / 2,    math.pi / 2]
# object_elevation_list = [0.0,   -math.pi / 4,   0.0,         -math.pi / 4,   0.0,            -math.pi / 4]
# azimuth_list = object_azimuth_list # just for dump file name
# elevation_list = object_elevation_list # just for dump file name


#test camera transform
dumproot = os.path.join(root_path, r'framework', r'modules', r'projection', r'camera_transform_test')
make_dir(dumproot)
camera_azimuth_list = [0.0,   0.0,            math.pi/4,   math.pi/4,      math.pi / 2,    math.pi / 2]
camera_elevation_list = [0.0,   -math.pi / 4,   0.0,         -math.pi / 4,   0.0,            -math.pi / 4]
azimuth_list = camera_azimuth_list # just for dump file name
elevation_list = camera_elevation_list # just for dump file name
object_azimuth_list = [0.0] * batchsize
object_elevation_list = [0.0] * batchsize


with tf.Session() as sess:
    # get camera transform matrix
    _camera_batch_azimuth = tf.constant(camera_azimuth_list, dtype=tf.float32)
    _camera_batch_elevation = tf.constant(camera_elevation_list, dtype=tf.float32)
    _camera_batch_tilt = tf.zeros_like(_camera_batch_azimuth)
    _camera_r = tfg.geometry.transformation.rotation_matrix_3d.from_euler(
        tf.stack([_camera_batch_elevation, _camera_batch_azimuth, _camera_batch_tilt], axis = -1)
    )
    _camera_t = tf.constant([[0,0,1]], dtype=tf.float32)
    _camera_batch_distance = tf.constant([1.5], dtype=tf.float32)
    _camera_t = _camera_t * _camera_batch_distance

    # get object transform matrix
    _object_batch_azimuth = tf.constant(object_azimuth_list, dtype=tf.float32)
    _object_batch_elevation = tf.constant(object_elevation_list, dtype=tf.float32)
    _object_batch_tilt = tf.zeros_like(_object_batch_azimuth)
    _object_r = tfg.geometry.transformation.rotation_matrix_3d.from_euler(
        tf.stack([_object_batch_elevation, _object_batch_azimuth, _object_batch_tilt], axis = -1)
    )
    _object_batch_distance = tf.constant([0.0], dtype=tf.float32)
    _object_t = tf.constant([[0, 0, 1]], dtype=tf.float32)
    
    _object_t = _object_t * _object_batch_distance

    _viewport = [-2, 2, -2, 2]

    _voxel = tf.placeholder(tf.float32, shape=(batchsize, 32, 32, 32, 1))

    if test_mod == 'perspective':
        camera_fov_radian = math.pi / 2
        camera_focal_pixel = image_size[0] / (2.0 * tf.tan(tf.constant(camera_fov_radian) / 2.0))
        camera_focal_pixel = tf.expand_dims(camera_focal_pixel, axis=0)
    else:
        camera_focal_pixel = None

    # print(_voxel.get_shape().as_list())
    # print(_camera_r.get_shape().as_list())
    # print(_camera_t.get_shape().as_list())
    # input()

    _batch_image = render_silouette(_voxel, None, None, _camera_r, _camera_t, focal_in_pixel=camera_focal_pixel, viewport = _viewport, z_near=0.0, z_far=2.5, z_sample_cnt=64, image_size=image_size)
    img = sess.run(_batch_image, feed_dict = {_voxel:voxel})
    # print(coord[0])
    # print('voxshape:{}'.format(vox.shape))
    # print('imgshape: {}'.format(img.shape))
    # input()

    # print(vox.shape)
    for i in range(batchsize):
        save_pfm(os.path.join(dumproot, r'result_{}_azimuth_{}pi_elev_{}pi.png'.format(test_mod, azimuth_list[i] / math.pi, elevation_list[i] / math.pi)),
                 img[i, ..., 0] * 255)
        print('Done')