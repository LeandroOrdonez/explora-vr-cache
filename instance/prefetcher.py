import os, re, pickle, sys, json
import requests
sys.path.append(r'./instance/')
sys.path.append(r'./util/')
import pandas as pd
import numpy as np
from threading import Thread
from config import Config
from redis_queue import RedisQueue
from redis import Redis
from segment_rank import SegmentRank

class Prefetcher:

    def __init__(self):
        print(f'[Prefetcher] __init__')
        self.buffer = Redis(host=os.getenv("REDIS_HOST"), db=0)
        self.init_buffer = Redis(host=os.getenv("REDIS_HOST"), db=1) # It buffers the first segments of all videos in the catalog and keeps them in memory
        self.buffer_keys = RedisQueue('buffer_keys', host=os.getenv("REDIS_HOST"), db=3)
        self.init_buffer_keys = RedisQueue('init_buffer_keys', host=os.getenv("REDIS_HOST"), db=3)
        self.buffer_keys_len = int(os.getenv("BUFFER_SIZE"))
        # self.n_queries = 0
        # self.n_hits = 0
        self.session = requests.Session()
        for video in Config.VIDEO_CATALOG:
            # args = (Config.T_HOR, Config.T_VERT, video, Config.SUPPORTED_QUALITIES[-1], 'seg_dash_trackX_X.m4s') # Only important elements from this tuple at initialization are T_VER, T_HOR and the video ID, the rest are placeholders, basically 
            Thread(
                target=self.prefetch_segment_into_init_buffer,
                args=(int(video), 1, 0),
                daemon=True
            ).start()
        if os.getenv('PERFECT_PREDICTION') == 'true':
            self.user_traces = self.load_user_traces()
        else:
            # self.prefetch_models = {}
            # for v in Config.VIDEO_CATALOG:	
	        #         self.prefetch_models[f'v{v}'] = {}	
	        #         for fold in range(1,9):	
	        #             self.prefetch_models[f'v{v}'][f'f{fold}'] = pickle.load(open(f'./instance/model_files/fold_{fold}/em_v{v}_th90.pkl','rb'))
            self.segment_rankings = dict()
        print(f'[Prefetcher] OK, Listening...')

    def prefetch_segment_into_init_buffer(self, video, segment, tile):
        # self.init_buffer[(video, segment)] = dict()
        self.init_buffer_keys.put(f'{video}:{segment}')
        for i_tile in range(Config.T_VERT*Config.T_HOR):
            for q_index in Config.SUPPORTED_QUALITIES:
                # print(f"PREFETCHING TILE {i_tile + 1} Q{q_index} FROM SEGMENT {segment} VIDEO {video}")
                Thread(
                    target=self.fetch_tile,
                    args=(i_tile + 1, segment, video, q_index, True),
                    daemon=True
                ).start()

    def prefetch_segment(self, video, segment, qualities, order):
        # print(f"[prefetch_segment] {args}")
        if self.buffer_keys.qsize() >= self.buffer_keys_len:
            first_key = self.buffer_keys.get().decode('utf-8')
            self.remove_segment_from_buffer(first_key)
            del self.segment_rankings[tuple(map(int, first_key.split(':')))]
        # self.buffer[(video, actual_segment + 1)] = dict()
        self.buffer_keys.put(f'{video}:{segment}')
        if (video, segment) not in self.segment_rankings:
            s_rank = SegmentRank()
            s_rank.update_rank(order)
            self.segment_rankings[(video, segment)] = s_rank
        for t_id in order:
            quality = qualities[t_id - 2]
            q_idx = int(1.5*quality**2 - 0.5*quality)
            Thread(
                target=self.fetch_tile,
                args=(t_id - 1, segment, video, q_idx),
                daemon=True
            ).start()                
            # print(f'[prefetch] Current buffer: {[(k, v.items()) for k, v in self.buffer.items()]}')

    def update_segment(self, video, segment, order):
        new_rank, kt_dist_accum, n_views = self.segment_rankings[(video, segment)].update_rank(order)
        vp_size = int(np.ceil(Config.T_VERT*Config.T_HOR*(1 - (kt_dist_accum/n_views))))
        for t_id in new_rank[:vp_size]:
            for q_index in Config.SUPPORTED_QUALITIES:
                Thread(
                    target=self.fetch_tile,
                    args=(t_id - 1, segment, video, q_index),
                    daemon=True
                ).start()

    def fetch_tile(self, tile, segment, video, quality, into_init_buffer=False):
        # print(f"[fetch_tile] seg_dash_track{tile}_{segment}.m4s")
        filename = f'seg_dash_track{tile}_{segment}.m4s'
        key = f'{video}:{segment}:{tile}:{quality}'
        # Redis.setnx (set if not exists) could've been used next but 
        # then the fetch call would always be made which is precisely
        # what we are trying to avoid
        if into_init_buffer and not self.init_buffer.exists(key):
            return self.init_buffer.set(key, self.fetch(Config.T_HOR, Config.T_VERT, video, quality, filename))
        elif not self.buffer.exists(key):
            return self.buffer.set(key, self.fetch(Config.T_HOR, Config.T_VERT, video, quality, filename))

    def fetch(self, t_hor, t_vert, video_id, quality, filename):
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
        query_string = f'prefetch={os.getenv("ENABLE_PREFETCHING") == "true"}&perfect_prediction={os.getenv("PERFECT_PREDICTION") == "true"}'
        tile_bytes = self.session.get(f'{url}?{query_string}').content
        return tile_bytes

    def remove_segment_from_buffer(self, key):
        keys = self.buffer.keys(f'{key}:*')
        self.buffer.delete(*keys)

def main():
    """ main method """

    # Create an instance of type Prefetcher
    p = Prefetcher()
    
    # Prepare subscriber
    redis_conn = Redis(host=os.getenv("REDIS_HOST"), charset="utf-8", decode_responses=True)
    pubsub = redis_conn.pubsub()
    pubsub.subscribe("prefetch")
    for message in pubsub.listen():
        if message.get("type") == "message":
            p_data = json.loads(message.get("data"))
            print(f"[Prefetcher] received: {p_data}")
            s_id = p_data['s_id']
            v_id = p_data['v_id']
            qualities = p_data['qualities']
            order = p_data['order']
            if not (p.buffer_keys.contains(f'{v_id}:{s_id}') or p.init_buffer_keys.contains(f'{v_id}:{s_id}')): #(int(os.getenv("VIEWPORT_SIZE")) > 1):
                Thread(
                    target=p.prefetch_segment,
                    args=(v_id, s_id, qualities, order),
                    daemon=True
                ).start()
            elif p.buffer_keys.contains(f'{v_id}:{s_id}'):
                Thread(
                    target=p.update_segment,
                    args=(v_id, s_id, order),
                    daemon=True
                ).start()



if __name__ == "__main__":
    main()