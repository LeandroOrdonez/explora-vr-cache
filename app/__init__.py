# app/__init__.py
import os
import logging
import traceback
import json
from flask_api import FlaskAPI
from flask import request, jsonify, abort, send_file, send_from_directory, safe_join
from util.file_handler_with_header import FileHandlerWithHeader as FileHandler
from io import BytesIO
import re


# local import
from instance.config import app_config
from instance.prefecth_handler import PrefetchBufferHandler

api_endpoint = ''

# Set logger
log_header = 'date|video|segment|tile|quality|run_time'
logger = logging.getLogger(__name__)
file_handler = FileHandler(filename=os.getenv('QUERY_LOG'), header=log_header, delay=True)
formatter = logging.Formatter('%(asctime)s|%(message)s')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

def create_app(config_name):
    from app.controllers import QHandler

    app = FlaskAPI(__name__, instance_relative_config=True, static_url_path='/static')
    app.config.from_object(app_config[config_name])
    app.config.from_pyfile('config.py')
    t_vert = app.config['T_VERT']
    t_hor = app.config['T_HOR']

    # /api/video/0/segment/0/tile/0?quality=5
    @app.route(f'{api_endpoint}/<string:video_id>/{t_hor}x{t_vert}/<int:quality>/<string:filename>', methods=['GET'])
    def get_tile(video_id, quality, filename):
        print("[get_tile] method call")
        try:
            if int(quality) not in app.config["SUPPORTED_QUALITIES"]:
                raise ValueError
            # print(f'quality={quality}')
            directory = f'{app.config["VIDEO_FILES_PATH"]}/{video_id}/{t_hor}x{t_vert}/{quality}'
            print(f'directory={directory}')
            # m = re.search(r'track(.*)_(.*)\.m4s',tile_name)
            # filename = f'seg_dash_track{tile_id}_{segment_id}.m4s'
            print(f'filename={filename}')
            filepath = os.path.join(app.root_path, directory, filename)
            tile_bytes = QHandler.get_video_tile(filepath)
            # print('Sending File...')
            # return send_from_directory(directory, filename=filename)
            return send_file(tile_bytes, mimetype='video/iso.segment')
        except Exception as e:
            print('[ERROR]', e)
            traceback.print_exc()
            if type(e) == ValueError:
                response = jsonify(error=f'Quality {quality} not in {app.config["SUPPORTED_QUALITIES"]}')
            else:
                response = jsonify(error=f"Requested (video, segment, tile) not found")
            response.status_code = 404   
            abort(response)

    return app
