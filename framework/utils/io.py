import os, sys
from io import StringIO as cStringIO
import numpy as np
import cv2

from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser

def make_dir(folder):
    if(os.path.exists(folder) == False):
        os.makedirs(folder)

def save_pfm(filepath, img, reverse = 1):
    color = None
    file = open(filepath, 'wb')
    if(img.dtype.name != 'float32'):
        img = img.astype(np.float32)

    color = True if (len(img.shape) == 3) else False

    if(reverse and color):
        img = img[:,:,::-1]

    img = img[::-1,...]

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (img.shape[1], img.shape[0]))

    endian = img.dtype.byteorder
    scale = 1.0
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)
    img.tofile(file)
    file.close()

def load_pfm(filepath, reverse = 1):
    file = open(filepath, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    color = (header == b'PF')

    width, height = map(int, file.readline().strip().decode('ascii').split(' '))
    scale = float(file.readline().rstrip().decode('ascii'))
    endian = '<' if(scale < 0) else '>'
    scale = abs(scale)

    rawdata = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()

    if(color):
        return rawdata.reshape(shape).astype(np.float32)[::-1,:,::-1]
    else:
        return rawdata.reshape(shape).astype(np.float32)[::-1,:]

def write_vox(filename, voxel, voxel_order = 'dhw', thereshold = 0.5):
    #transpose to match orders
    # if(voxel_order == 'dhw'):
    #     _voxel_data = voxel.transpose([2,1,0])
    #     _voxel_data = _voxel_data[::-1,...]
    _voxel_data = voxel >= thereshold
    _voxel_data = _voxel_data[:,::-1,:]
    vox = Vox.from_dense(_voxel_data)
    VoxWriter(filename, vox).write()


def load_vox(filename):
    _voxel_data = VoxParser(filename).parse().to_dense()
    _voxel_data = _voxel_data[:,::-1,:]

    return _voxel_data