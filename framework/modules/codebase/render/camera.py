#camera util based on tensorflow graphics
import tensorflow as tf
import tensorflow_graphics as tfg

#NOTE: camera intrinsics should be in pixel unit
def generate_viewmap_camera_space(image_size, focal_in_pixel = None):
    if(focal_in_pixel == None):
        #ortho
        return tfg.rendering.camera.orthographic.ray(tf.zeros(image_size + [2]))
    else:
        #persp
        sensor_coord = tfg.geometry.representation.grid.generate((image_size[1] - 1.0, image_size[0] - 1.0), (0.0, 0.0), [image_size[1], image_size[0]]) # h w
        principal_point = tf.expand_dims(tf.expand_dims(tf.constant([image_size[1], image_size[0]], dtype=tf.float32), axis = 0), axis = 0) / 2 # (2, )
        focal = tf.stack([focal_in_pixel, focal_in_pixel], axis = -1)

        focal = tf.expand_dims(tf.expand_dims(focal, axis = -2), axis = -2)# 1 1 1 2
        if(len(focal.get_shape().as_list()) == 4):
            sensor_coord = tf.expand_dims(sensor_coord, axis = 0) # 1 h w
            principal_point = tf.expand_dims(principal_point, axis = 0)# 1 2

        return tfg.rendering.camera.perspective.ray(sensor_coord, focal, principal_point)