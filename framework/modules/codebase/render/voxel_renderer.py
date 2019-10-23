# camera class based on tensorflow graphics
import tensorflow as tf
import tensorflow_graphics as tfg
from .camera import generate_viewmap_camera_space
from ..math import interpn_batch, apply_rotate
from ..layers import denormalizeInput


def object_coord_to_data_coord(object_coord, cube_size):
    #assume the object voxel are occupied volume of [-1, 1]^3
    #cube_size is a int number. TODO: support voxel data with non-cube size
    #the data order of voxel shape are z-y-x, from small to large
    #the object coord are of shape [..., 3] where the last dimension is [x,y,z] coord

    _x, _y, _z = tf.unstack(object_coord, axis=-1)
    _mask_x = tf.logical_and(
        tf.greater_equal(_x, -1),
        tf.less_equal(_x, 1))
    _mask_y = tf.logical_and(
        tf.greater_equal(_y, -1),
        tf.less_equal(_y, 1))
    _mask_z = tf.logical_and(
        tf.greater_equal(_z, -1),
        tf.less_equal(_z, 1))
    _mask = tf.expand_dims(tf.cast(tf.logical_and(tf.logical_and(_mask_x, _mask_y), _mask_z), tf.float32), axis = -1) 

    _x = denormalizeInput(_x) * (cube_size - 1)
    _y = denormalizeInput(_y) * (cube_size - 1)
    _z = (cube_size - 1) * (1 - denormalizeInput(_z))

    _coord = tf.stack([_z, _y, _x], axis=-1)

    return _coord, _mask

#drc-based voxel render
def resample_voxels(voxel_data, 
                    object_r, object_t,
                    camera_r, camera_t,
                    focal_in_pixel, viewport,
                    z_near, z_far,
                    z_sample_cnt, image_size):
    _new_dim = 1 if(len(voxel_data.get_shape().as_list()) == 5) else 0
    batch_size = voxel_data.get_shape().as_list()[0]
    z_sample = tf.linspace(-z_near, -z_far, z_sample_cnt)
    # generate sample points along ray
    if (focal_in_pixel == None):
        # assume ortho view volume is 2D viewport with [-z_near, -z_far] in camera space
        xl, xr, yl, yr = viewport        
        ray_sampled_points_camera_space = tfg.geometry.representation.grid.generate((xl, yl, -z_near),
                                                                                    (xr, yr, -z_far),
                                                                                    [image_size[1], image_size[0], z_sample_cnt])   #w*h*d*3
        ray_sampled_points_camera_space = tf.expand_dims(ray_sampled_points_camera_space, 0)  # 1whd3
    else:
        # compute ray
        _viewdir_camera = generate_viewmap_camera_space(image_size, focal_in_pixel)         #w*h*3 (n*w*h*3)
        _viewdir_camera = tf.expand_dims(_viewdir_camera, -2)                               #w*h*1*3 (n*w*h*1*3)
        z_sample = tf.expand_dims(tf.expand_dims(tf.expand_dims(z_sample, -1), 0), 0)       #1*1*d*1
        if (_new_dim):
            z_sample = tf.expand_dims(z_sample, 0)                                          #1*1*1*d*1

        ray_sampled_points_camera_space = _viewdir_camera * z_sample                        # w*h*d*3 (n*w*h*zsample*3)

    # convert to world space by applying inverse camera transform
    ray_sampled_points_world_space = ray_sampled_points_camera_space
    if(camera_t != None):
        camera_t = tf.expand_dims(tf.expand_dims(tf.expand_dims(camera_t, _new_dim), _new_dim), _new_dim)
        ray_sampled_points_world_space = ray_sampled_points_world_space + camera_t

    if (camera_r != None):
        ray_sampled_points_world_space = apply_rotate(ray_sampled_points_world_space, camera_r)
    # convert to object space by applying inverse object transform
    ray_sampled_points_object_space = ray_sampled_points_world_space
    if(object_t != None):
        object_t = tf.expand_dims(tf.expand_dims(tf.expand_dims(object_t, _new_dim), _new_dim), _new_dim)
        ray_sampled_points_object_space = ray_sampled_points_object_space - object_t

    if (object_r != None):
        inv_rotation = tfg.geometry.transformation.rotation_matrix_3d.inverse(object_r)
        ray_sampled_points_object_space = apply_rotate(ray_sampled_points_object_space, inv_rotation)

    #convert to z-y-x order
    if(_new_dim):
        ray_sampled_points_object_space = tf.transpose(ray_sampled_points_object_space, perm = [0,3,2,1,4])
    else:
        ray_sampled_points_object_space = tf.transpose(ray_sampled_points_object_space, perm = [2,1,0,3])

    # convert to vol coord (0, w/h/d)
    # if we extend the method with warp field like nerual volumes, it can be implemented here
    n, d, h, w, c = voxel_data.get_shape().as_list()
    _data_coord, _data_mask = object_coord_to_data_coord(ray_sampled_points_object_space, h)

    # sample voxel
    voxel_sampled = interpn_batch(voxel_data, _data_coord, interp_method='linear')
    voxel_sampled = voxel_sampled * _data_mask

    return voxel_sampled

def compute_ray_potentials(voxel_occ_prob):
    # compute ray potentials
    _prod_axis = 1 if(len(voxel_occ_prob.get_shape().as_list()) == 5) else 0
    inv_cum_prod = tf.cumprod(1.0 - voxel_occ_prob + 1e-12, axis=_prod_axis, exclusive=True)#, reverse=True)
    ray_potentials = tf.multiply(voxel_occ_prob, inv_cum_prod)

    return ray_potentials

def compute_image_drc(ray_potentials, output = 'silouette', **kwargs_additional):
    _prod_axis = 1 if(len(ray_potentials.get_shape().as_list()) == 5) else 0
    if(output == 'silouette'):
        # to silouette
        voxel_image = tf.reduce_sum(ray_potentials, axis=_prod_axis)  # N*H*W*C
    elif(output == 'depth'):
        voxel_z = kwargs_additional['voxel_z']
        voxel_image = tf.reduce_sum(voxel_z * ray_potentials, axis=_prod_axis)

    if(_prod_axis == 1):
        voxel_image = voxel_image[:,::-1,...]
    else:
        voxel_image = voxel_image[::-1,...]
    return voxel_image

def compute_image_transparent_v1(voxel_alpha, voxel_data = None):
    #voxel_alpha: [...,1]
    _prod_axis = 1 if(len(voxel_alpha.get_shape().as_list()) == 5) else 0
    #compute cutoff alpha
    alpha_sum = tf.cumsum(voxel_alpha, axis=_prod_axis)
    alpha_cutoff = tf.where(
        tf.greater(alpha_sum, 1.0),
        tf.zeros_like(voxel_alpha),
        voxel_alpha
    )
    #compute image
    alpha_image = tf.reduce_sum(alpha_cutoff, axis = _prod_axis)
    if(_prod_axis == 1):
        alpha_image = alpha_image[:,::-1,...]
    else:
        alpha_image = alpha_image[::-1,...]

    voxel_image = None
    if(voxel_data != None):
        voxel_image = tf.reduce_sum(alpha_cutoff * voxel_data, axis = _prod_axis)
        if(_prod_axis == 1):
            voxel_image = voxel_image[:,::-1,...]
        else:
            voxel_image = voxel_image[::-1,...]
    
    return voxel_image, alpha_image    

def render_silouette_drc(voxel_data,
                     object_r, object_t,
                     camera_r, camera_t,
                     focal_in_pixel, viewport,
                     z_near, z_far,
                     z_sample_cnt, image_size):
    with tf.name_scope('render_silouette_drc'):
        voxel_resample = resample_voxels(voxel_data, object_r, object_t, camera_r, camera_t, focal_in_pixel, viewport, z_near, z_far, z_sample_cnt, image_size)
        ray_potentials = compute_ray_potentials(voxel_resample)
        silouette_image = compute_image_drc(ray_potentials, 'silouette')
    
    return silouette_image

def render_silouette_nv(voxel_data,
                     object_r, object_t,
                     camera_r, camera_t,
                     focal_in_pixel, viewport,
                     z_near, z_far,
                     z_sample_cnt, image_size):
    with tf.name_scope('render_silouette_transparent'):
        voxel_resample = resample_voxels(voxel_data, object_r, object_t, camera_r, camera_t, focal_in_pixel, viewport, z_near, z_far, z_sample_cnt, image_size)
        _, silouette_image = compute_image_transparent_v1(voxel_resample)
    
    return silouette_image