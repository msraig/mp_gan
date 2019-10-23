import json, math
import tensorflow as tf
import tensorflow_graphics as tfg

from ...utils.log import log_message
from ..codebase.render.voxel_renderer import render_silouette_drc, render_silouette_nv
from ..module_base import module_base
from ..codebase.visualization import tileImage

class identity_projection(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        generator = kwargs['generator']

        global_data_dict = kwargs['global_data_dict']

        if(phase == 'train'):
            gpu_list = cfg_args.gpu_list
        else:
            gpu_list = [0]

        log_message('identity_projection', '---Building subgraph...---')
        tf_output_all_gpu = []        
        
        with tf.name_scope('projection'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    _voxel = generator.public_ops['voxel_generated'][g_id]
                    tf_output_all_gpu.append(_voxel)

        self.public_ops['projected_voxel'] = tf_output_all_gpu


class multi_view_mask_projection(module_base):
    def __init__(self):
        module_base.__init__(self)

    def build_graph(self, **kwargs):
        cfg_args = kwargs['cfg_args']
        phase = kwargs['phase']
        generator = kwargs['generator']
        
        global_data_dict = kwargs['global_data_dict']

        if(phase == 'train'):
            data_loader = kwargs['data_loader']

        gpu_list = cfg_args.gpu_list

        #load view definiation
        with open(cfg_args.view_def_file, 'r') as f:
            raw_view_data = json.load(f)
        #azimuth:
        #elevation:
        n_view_bin = raw_view_data['n_view_bin']
        global_data_dict['n_view_bin'] = n_view_bin

        #extrinsic
        azimuth_ranges = math.pi / 180.0 * tf.constant(raw_view_data['azimuth_ranges'], dtype=tf.float32)    #k*2
        elevation_ranges = math.pi / 180.0 * tf.constant(raw_view_data['elevation_ranges'], dtype=tf.float32)
        
        camera_dist = raw_view_data['camera_dist']
        
        #intrinsic
        camera_type = cfg_args.camera.type
        camera_fov_radian = cfg_args.camera.fov_degree * math.pi / 180.0
        
        image_size = cfg_args.camera.image_size
        z_near = cfg_args.camera.z_near
        z_far = cfg_args.camera.z_far
        z_sample_cnt = cfg_args.camera.z_sample_cnt
        
        if(camera_type == 'persp'):
            camera_focal_pixel = image_size[0] / (2.0 * tf.tan(tf.constant(camera_fov_radian) / 2.0))
            camera_focal_pixel = tf.expand_dims(camera_focal_pixel, axis = 0)
            _viewport = None
        else:
            camera_focal_pixel = None
            _viewport = cfg_args.camera.viewport

        log_message('multi_view_mask_projection', '---Building subgraph...---')
        tf_image_all_gpu = []
        tf_view_label_all_gpu = []
        tf_azimuth_all_gpu = []
        tf_elevation_all_gpu = []

        #select render methods
        if(cfg_args.render_method == 'drc'):
            log_message('multi_view_mask_projection', '---Use DRC render...---')
            render_func = render_silouette_drc
        elif(cfg_args.render_method == 'nv'):
            log_message('multi_view_mask_projection', '---Use NV render...---')
            render_func = render_silouette_nv
        else:
            raise NotImplementedError("Undefined method!")

        with tf.name_scope('projection'):
            for g_id in range(0, len(gpu_list)):
                with tf.device('/gpu:{}'.format(g_id)):
                    _voxel = generator.public_ops['voxel_generated'][g_id]
                    if(phase == 'train'):
                        _view_label = data_loader.public_ops['input_view_label'][g_id]   #(n,)                        
                    else:
                        _view_label = tf.random_uniform(shape=[_voxel.get_shape().as_list()[0]], minval = 0, maxval = n_view_bin, dtype=tf.int32)

                    _camera_r, _camera_t, _batch_azimuth, _batch_elevation = _get_camera_mat(_view_label, azimuth_ranges, elevation_ranges, camera_dist)
                    _batch_image = render_func(_voxel, None, None, _camera_r, _camera_t, focal_in_pixel=camera_focal_pixel, viewport=_viewport, z_near=z_near, z_far=z_far, z_sample_cnt=z_sample_cnt, image_size=image_size)

                    tf_image_all_gpu.append(_batch_image)
                    tf_view_label_all_gpu.append(_view_label)

                    tf_azimuth_all_gpu.append(_batch_azimuth)
                    tf_elevation_all_gpu.append(_batch_elevation)

        if(phase != 'test'):
            all_image = tf.concat(tf_image_all_gpu, axis=0)
            vis_input_data = tileImage(all_image, nCol=8)
            self.add_summary(vis_input_data, phase, 'image', 'projected_mask')        

        self.public_ops['projected_mask'] = tf_image_all_gpu
        self.public_ops['view_label'] = tf_view_label_all_gpu
        self.public_ops['azimuth'] = tf_azimuth_all_gpu
        self.public_ops['elevation'] = tf_elevation_all_gpu

def _get_camera_mat(_view_label, azimuth_ranges, elevation_ranges, camera_dist):
    _random_vector = tf.random_uniform(shape=_view_label.get_shape().as_list())
    _batch_azimuth_minmax = tf.gather(azimuth_ranges, _view_label)
    _batch_elevation_minmax = tf.gather(elevation_ranges, _view_label)

    _batch_azimuth = _random_vector * (_batch_azimuth_minmax[...,1] - _batch_azimuth_minmax[...,0]) + _batch_azimuth_minmax[...,0]
    _batch_elevation = _random_vector * (_batch_elevation_minmax[...,1] - _batch_elevation_minmax[...,0]) + _batch_elevation_minmax[...,0]
    _batch_tilt = tf.zeros_like(_batch_azimuth)

    _camera_t = tf.constant([[0,0,camera_dist]], dtype=tf.float32)
    _camera_r = tfg.geometry.transformation.rotation_matrix_3d.from_euler(
        tf.stack([_batch_elevation, _batch_azimuth, _batch_tilt], axis = -1)
    )

    return _camera_r, _camera_t, _batch_azimuth, _batch_elevation
                




        
