import os
import functools
import glob
import csv
from pathlib import Path
from collections import namedtuple
import raw_cache
import copy


import pandas as pd
from dotenv import load_dotenv
import SimpleITK as sitk
import numpy as np
import torch

from util import irc_to_xyz, xyz_to_irc, XyzTuple


load_dotenv()
data_dir = Path(os.environ['DataDir'])
candidates_data = data_dir / 'candidates.csv'
annotations_data = data_dir / 'annotations.csv'

CandidateInfoTuple = namedtuple(
 'CandidateInfoTuple',
 'isNodule_bool, diameter_mm, series_uid, center_xyz',
)


def annotations_to_dict(csv_file: str) -> dict:
    """
    Returns a dictionary with DICOM ids as keys and a
    list of tuple values in the form ((x_coord, y_coord, z_coord), diameter)
    Param:
        csv_file: path as string or path object
    Return:
        dictionary in the for
        {
        DICOM1: [((x1,y1,z1), diam1),((x2, y2, z2), diam2) ...],
        DICOM2: [], ...

        }
    """

    df_annotations = pd.read_csv(annotations_data)
    df_annotations['coords'] = tuple(
        zip(
            df_annotations.coordX, df_annotations.coordY,
            df_annotations.coordZ
            )
        )
    df_annotations['coords_diam'] = tuple(
        zip(df_annotations.coords, df_annotations.diameter_mm)
        )
    annotations_dict = df_annotations.groupby(
        'seriesuid')['coords_diam'].apply(list).to_dict()
    return annotations_dict


@functools.lru_cache(1)
def get_candidate_info_list(requireOnDisk_bool=True):
    mhd_list = glob.glob('data/src/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    presentOnDisk_set
    diameter_dict = annotations_to_dict(annotations_data)
    candidateInfo_list = get_candidates(
        presentOnDisk_set, requireOnDisk_bool, diameter_dict
        )
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


def get_candidates(
    presentOnDisk_set: set, requireOnDisk_bool: bool, diameter_dict:  dict
) -> list:
    candidateInfo_list = []
    with open(candidates_data, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(
                        candidateCenter_xyz[i] - annotationCenter_xyz[i]
                        )
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                    else:
                        candidateDiameter_mm = annotationDiameter_mm
                        break

            candidateInfo_list.append(CandidateInfoTuple(
              isNodule_bool,
              candidateDiameter_mm,
              series_uid,
              candidateCenter_xyz,
            ))
    return candidateInfo_list


@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(
    cand_series_uid, cand_center_xyz, widh_irc
):
    ct = get_ct(cand_series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(cand_center_xyz, widh_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(
        self, val_stride=0, isValSet_bool=None, series_uid=None,
    ):
        self.candidate_info_list = copy.copy(get_candidate_info_list())
        if series_uid:
            self.candidate_info_list = filter(
                lambda x: x.series_uid == series_uid, self.candidate_info_list
                )
            self.candidate_info_list = list(self.candidate_info_list)
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, index):
        candidateInfo_tup = self.candidate_info_list[index]
        width_irc = (32, 48, 48)
        candidate_array, center_irc = get_ct_raw_candidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )
        candidate_tensor = torch.from_numpy(candidate_array)
        candidate_tensor = candidate_tensor.to(torch.float32)
        candidate_tensor = candidate_tensor.unsqueeze(0)
        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
            ], dtype=torch.long)
        return (
            candidate_tensor,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )


class Ct:
    def __init__(self, series_uid):
        mhd_glob = glob.glob(f'{data_dir}/subset*/{series_uid}.mhd')[0]
        # python simpleITK is a wrapper around a C++ lib using SWIG. The
        # return is a swig object
        ct_mhd = sitk.ReadImage(mhd_glob)
        ct_array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # bound HU values between -1000 (air) and 1000 (roughly bone density)
        ct_array.clip(-1000, 1000, ct_array)

        self.series_uid = series_uid
        self.ct_hu_array = ct_array
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.voxel_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_array = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz_to_irc(
            center_xyz,
            self.origin_xyz,
            self.voxel_size_xyz,
            self.direction_array,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_index = int(round(center_val - width_irc[axis]/2))
            end_index = int(start_index + width_irc[axis])
            slice_list.append(slice(start_index, end_index))

        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc
