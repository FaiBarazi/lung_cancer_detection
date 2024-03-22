from collections import namedtuple
import numpy as np


IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc_to_xyz(
    coord_irc: list, origin_xyz: list, voxel_size_xyz: list, direction_a: list
) -> XyzTuple:
    """
    Convert IRC (index, row, column) coordinates of CT scan image
    to XYZ coordinates.
    """
    # coord_irc to numpy array and flip the coordinates to CRI
    # to match xYZ order.
    cri_array = np.array(coord_irc)[::-1]
    origin_array = np.array(origin_xyz)
    xyz_array = np.array(origin_xyz)
    voxel_size_xyz = np.array(voxel_size_xyz)
    # Scale cri by the voxel size.
    cri_scaled = cri_array * voxel_size_xyz
    # Linear transformation: cri by the direction matrix and
    # shift by the origin.
    coords_xyz = (direction_a @ cri_scaled) + origin_array
    return XyzTuple(*coords_xyz)
