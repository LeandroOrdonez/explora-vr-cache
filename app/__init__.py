# app/__init__.py
import os
import logging
import traceback
import json
from flask_api import FlaskAPI
from flask import request, jsonify, abort, send_file, send_from_directory, safe_join, make_response  
from util.file_handler_with_header import FileHandlerWithHeader as FileHandler
from io import BytesIO
import re


# local import
from instance.config import app_config

api_endpoint = ''

# Set logger
# log_header = 'date|video|quality|filename|k|prefetch|perfect_prediction'
# logger = logging.getLogger(__name__)
# file_handler = FileHandler(filename=os.getenv('QUERY_LOG'), header=log_header, delay=True)
# formatter = logging.Formatter('%(asctime)s|%(message)s')
# file_handler.setFormatter(formatter)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)
# logger.setLevel(logging.DEBUG)

def create_app(config_name):
    from app.controllers import QHandler

    app = FlaskAPI(__name__, instance_relative_config=True, static_url_path='/static')
    app.config.from_object(app_config[config_name])
    app.config.from_pyfile('config.py')
    t_vert = app.config['T_VERT']
    t_hor = app.config['T_HOR']

    # /0/4x4/4/seg_dash_track1_30.m4s
    @app.route(f'{api_endpoint}/<string:video_id>/{t_hor}x{t_vert}/<int:quality>/<string:filename>', methods=['GET'])
    def get_tile(video_id, quality, filename):
        print("[get_tile] method call")
        try:
            if int(quality) not in app.config["SUPPORTED_QUALITIES"]:
                raise ValueError(f'Quality {quality} not in {app.config["SUPPORTED_QUALITIES"]}')
            # logger.info(f'{video_id}|{quality}|{filename}|{request.args.get("k")}|{request.args.get("prefetch")}|{request.args.get("perfect_prediction")}')
            user_id = int(request.args.get("user_id")) if request.args.get("user_id") else None
            vp_size = int(request.args.get("k")) if request.args.get("k") else -1
            fold = int(request.args.get("fold")) if request.args.get("fold") else 1
            # Is this the center tile ?
            ct = bool(request.args.get("ct"))
            # if vp_size < 0 or vp_size > t_hor*t_vert:
            #     raise ValueError(f'Viewport size value ({vp_size}) is not valid (0 < vp size <= {t_hor*t_vert})')
            quality_upgrade, tile_bytes = QHandler.get_video_tile(t_hor, t_vert, video_id, quality, filename, vp_size, user_id, fold, ct)
            # print('Sending File...')
            # return send_from_directory(directory, filename=filename)
            response = make_response(send_file(BytesIO(tile_bytes), mimetype='video/iso.segment'))
            response.headers['X-Quality-Upgrade'] = 'true' if quality_upgrade else 'false'
            return response

        except Exception as e:
            print('[ERROR]', e)
            traceback.print_exc()
            if type(e) == ValueError:
                response = jsonify(error=e)
            else:
                response = jsonify(error=f"Requested (video, segment, tile) not found")
            response.status_code = 404   
            abort(response)

    return app
