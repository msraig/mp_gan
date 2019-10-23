import tensorflow as tf
import tensorflow_graphics as tfg

import itertools
import numpy as np

# replace tfg's rotate method
# tfg's rotate have issues handling rotate a lot of points.
# see issues here: https://github.com/tensorflow/graphics/issues/26

def apply_rotate(points, mat, name = None):
    with tf.name_scope(name, "rotation_matrix_3d_rotate"):       
        if(len(mat.get_shape()) == 2):
            mat_batch = tf.expand_dims(mat, axis = 0)
        else:
            mat_batch = mat
            
        if(len(points.get_shape()) == 1):
            points_batch = tf.expand_dims(points, axis = 0)
        else:
            points_batch = points
        #flatten shapes
        input_shape = points_batch.get_shape()
        n = input_shape[0]
        points_flatten = tf.reshape(points_batch, [input_shape[0],-1,3])
        flatten_shape = points_flatten.get_shape()
        m = mat_batch.get_shape().as_list()[0]

        # if(m == 1 and n != 1):
        #     mat_batch = tf.tile(mat_batch, [n, 1, 1])
        # if(n == 1 and m != 1):
        #     points_flatten = tf.tile(points_flatten, [m, 1, 1])

        #see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/4tgsOSxwtkY for here's batch matmul trick
        #input:       [n, k, 3]
        #mat_batch:   [n, 3, 3]
        _input_transposed = tf.transpose(points_flatten, perm = [0, 2, 1])		#[n, 3, k]
        _points_xform = tf.matmul(mat_batch, _input_transposed)					#[n, 3, k]
        _points_output = tf.transpose(_points_xform, perm = [0, 2, 1])          #[n, k, 3]

        return tf.reshape(_points_output, [max(m, n)] + points.get_shape().as_list()[1::])


def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def interpn_batch(vol, loc, interp_method='linear'):
    #both vol and loc should have batch dimensions
    n_vol = vol.get_shape().as_list()[0]
    n_loc = loc.get_shape().as_list()[0]

    vol_list = tf.unstack(vol, axis = 0)
    loc_list = tf.unstack(loc, axis = 0)

    output_list = []
    if(n_loc == 1):
        for i in range(n_vol):
            output_list.append(interpn(vol_list[i], loc_list[0]))
    elif(n_vol == 1):
        for i in range(n_loc):
            output_list.append(interpn(vol_list[0], loc_list[i]))
    elif(n_loc != n_vol):
        raise Exception("Loc dimension and vol dimension does not match.")
    else:
        for i in range(n_vol):
            output_list.append(interpn(vol_list[i], loc_list[i]))
    
    return tf.stack(output_list, axis = 0)

#Code pieces from:
#https://github.com/voxelmorph/voxelmorph/blob/master/ext/neuron/neuron/utils.py
def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow
    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions
    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
    Returns:
        new interpolated volume of the same size as the entries in loc
    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """
    
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')
    
    if isinstance(vol.shape, (tf.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
        clipped_loc = [tf.clip_by_value(loc[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0] # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = tf.expand_dims(wt, -1)
            
            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
        
    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')

        # clip values
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx) 

    return interp_vol