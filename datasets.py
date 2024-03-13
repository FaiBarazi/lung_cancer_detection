import os
import functools
import glob
import csv
from pathlib import Path
from collections import namedtuple

import pandas as pd
from dotenv import load_dotenv


load_dotenv()
data_dir = Path(os.environ['DataDir'])
candidates_data = data_dir / 'candidates.csv'
annotations_data = data_dir / 'annotations.csv'

CandidateInfoTuple = namedtuple(
 'CandidateInfoTuple',
 'isNodule_bool, diameter_mm, series_uid, center_xyz',
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data/src/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    presentOnDisk_set
    annotations_dict = annotations_to_dict(annotations_data)


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

