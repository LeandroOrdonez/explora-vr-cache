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

class QHandler():
    """This class represents an entity in charge of handling and processing queries issued to the API."""    

    def __init__(self):
        """initialize."""

    @staticmethod
    @PrefetchBufferHandler
    def get_video_tile(filepath):
        print("[get_video_tile] method call")
        print(f'filepath = {filepath}')
        with open(filepath, 'rb') as fh:
            tile_bytes = BytesIO(fh.read())
        return tile_bytes