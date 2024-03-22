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
    voxel_size_xyz = np.array(voxel_size_xyz)
    # Scale cri by the voxel size.
    cri_scaled = cri_array * voxel_size_xyz
    # Linear transformation: cri by the direction matrix and
    # shift by the origin.
    coords_xyz = (direction_a @ cri_scaled) + origin_array
    return XyzTuple(*coords_xyz)


def xyz_to_irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a) -> XyzTuple:
    origin_array = np.array(origin_xyz)
    voxel_size_array = np.array(vxSize_xyz)
    coord_array = np.array(coord_xyz)
    cri_array = (
        (coord_array - origin_array) @ np.linalg.inv(direction_a)
        ) / voxel_size_array
    cri_array = np.round(cri_array)
    return IrcTuple(int(cri_array[2]), int(cri_array[1]), int(cri_array[0]))
