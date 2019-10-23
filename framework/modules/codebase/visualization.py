import tensorflow as tf
from .layers import bbox_normalize

def tileImage(input_batch, nCol = 10, pad_value = 1.0):
    input_shape = input_batch.get_shape().as_list()
    nImg = input_shape[0]
    nRow = (nImg - 1) // nCol + 1
    output_rows = []
    for r in range(0, nRow):
        output_row = []
        for c in range(0, nCol):
            if(r*nCol+c < nImg):
                output_row.append(input_batch[r*nCol+c,:,:,:])
            else:
                output_row.append(pad_value * tf.ones_like(input_batch[0]))
        output_rows.append(tf.concat(output_row, axis=1))
    output = tf.concat(output_rows, axis=0)
    return output[tf.newaxis,...]

def packVoxelData(voxel_pos, voxel_color = None, voxel_size = 0.5):
    #voxel color should in RGB        (B, N, 3)
    #voxel position should in XYZ    (B, N, 3)
    #all input are with leading batch dim
    config_dict = {
        'material': {
            'cls': 'PointsMaterial',
            'size': voxel_size
        }
    }

    #normalize voxel positions to [-1, 1]
    normalized_pos = bbox_normalize(voxel_pos)

    if(voxel_color == None):
        return dict(vertices = normalized_pos, config_dict = config_dict)
    else:
        return dict(vertices = normalized_pos, color = voxel_color)
