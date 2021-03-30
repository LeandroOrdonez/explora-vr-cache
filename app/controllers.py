from io import BytesIO
from instance.prefetch_handler import PrefetchBufferHandler
import numpy as np
import pandas as pd
import os
import json
import re
import time
import traceback
from instance.config import Config
from datetime import datetime
import requests 

class QHandler():
    """This class represents an entity in charge of handling and processing queries issued to the API.""" 

    def __init__(self):
        """initialize."""

    @staticmethod
    @PrefetchBufferHandler
    def get_video_tile(t_hor, t_vert, video_id, quality, filename, vp_size=4, user_id=None, fold=1):
        # print("[get_video_tile] method call")
        server_url = os.getenv("SERVER_URL") if os.getenv("SERVER_URL") else "http://localhost:5000"
        url = f'{server_url}/{video_id}/{t_hor}x{t_vert}/{quality}/{filename}'
        # print(f'directory={directory}')
        # m = re.search(r'track(.*)_(.*)\.m4s',tile_name)
        # filename = f'seg_dash_track{tile_id}_{segment_id}.m4s'
        # print(f'filename={filename}')
        # filepath = os.path.join(app_root_path, directory, filename)
        # print(f'filepath = {filepath}')
        # with open(filepath, 'rb') as fh:
        #     tile_bytes = BytesIO(fh.read())
        query_string = f'k={vp_size}&fold={fold}&prefetch={os.getenv("ENABLE_PREFETCHING") == "true"}&perfect_prediction={os.getenv("PERFECT_PREDICTION") == "true"}'
        tile_bytes = requests.get(f'{url}?{query_string}').content
        return tile_bytes