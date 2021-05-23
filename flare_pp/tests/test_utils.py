import os
import numpy as np
import pytest
import sys
from flare_pp.utils import add_sparse_indices_to_xyz

def test_utils():
    add_sparse_indices_to_xyz(
        xyz_file_in="all_dft_frames.xyz", 
        ind_file="all_select_idx.txt",
        xyz_file_out="dft_data.xyz",
    )
