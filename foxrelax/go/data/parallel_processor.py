# -*- coding:utf-8 -*-
import sys
import os
import tarfile
import gzip
import shutil
import glob
import multiprocessing
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.go.encoder.base import get_encoder_by_name
from foxrelax.go.data.index_processor import KGSIndex
from foxrelax.go.data.sampling import Sampler
from foxrelax.go.gosgf import Sgf_game
from foxrelax.go.goboard_fast import (Board, GameState, Move)
from foxrelax.go.gotypes import (Player, Point)