import os, re, pickle, sys, json
import requests
sys.path.append(r'./instance/')
sys.path.append(r'./util/')
import pandas as pd
import numpy as np
from time import perf_counter
from threading import Thread
from config import Config
from redis_queue import RedisQueue
from redis import Redis
from segment_rank import SegmentRank

class Prefetcher:

    def __init__(self):
        print(f'[Prefetcher] __init__')
        self.collective_buffer = Redis(host=os.getenv("REDIS_HOST"), db=0)
        self.init_buffer = Redis(host=os.getenv("REDIS_HOST"), db=1) # It buffers the first segments of all videos in the catalog and keeps them in memory
        self.buffer_keys = RedisQueue('buffer_keys', host=os.getenv("REDIS_HOST"), db=3)
        self.init_buffer_keys = RedisQueue('init_buffer_keys', host=os.getenv("REDIS_HOST"), db=3)
        self.user_buffer = Redis(host=os.getenv("REDIS_HOST"), db=4)
        self.buffer_keys_len = int(os.getenv("BUFFER_SIZE"))
        self.prefetch_mode = os.getenv("PREFETCH_MODE", 'auto') # 'auto', 'hq' or 'lq'. Default: 'auto'
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
                    target=self.buffer_tile_into_init_buffer,
                    args=(i_tile + 1, segment, video, q_index),
                    daemon=True
                ).start()

    def prefetch_segment(self, video, segment, user, qualities, order):
        # print(f"[prefetch_segment] {args}")
        # Let's start by updating the segment tiles in the collective buffer
        updated = self.update_segment(video, segment, qualities, order)
        # Update the keys from the collective buffer 
        if not self.buffer_keys.contains(f'{video}:{segment}'):
            if self.buffer_keys.qsize() >= self.buffer_keys_len:
                first_key = self.buffer_keys.get().decode('utf-8')
                self.remove_segment_from_collective_buffer(first_key)
            # self.buffer[(video, actual_segment + 1)] = dict()
            self.buffer_keys.put(f'{video}:{segment}')
        # Create a new rank object for the current video and segment
            s_rank = SegmentRank()
            s_rank.update_rank(order)
            self.segment_rankings[(video, segment)] = s_rank
        for t_id in order:
            if self.prefetch_mode == 'auto':
                quality = qualities[t_id - 2]
                q_idx = int(1.5*quality**2 - 0.5*quality)
            elif self.prefetch_mode == 'hq':
                q_idx = Config.SUPPORTED_QUALITIES[-1]
            else:
                q_idx = Config.SUPPORTED_QUALITIES[0]
            Thread(
                target=self.buffer_tile,
                args=(t_id - 1, segment, video, q_idx, user, not updated), # if updated => don't store tile into the collective buffer
                daemon=True
            ).start()
        # Remove tiles from the previous segment:
        # prev_seg_key = f'{user}:{video}:{segment - 1}'
        # Thread(
        #     target=self.remove_segment_from_user_buffer,
        #     args=(prev_seg_key,),
        #     daemon=True
        # ).start()
        return

    def update_segment(self, video, segment, qualities, order):
        if (video, segment) in self.segment_rankings:
            # start = perf_counter()
            new_rank, kt_dist_accum, n_views = self.segment_rankings[(video, segment)].update_rank(order)
            # print(f'{perf_counter() - start},{video}') # Report the time it takes to update the segment rank
            vp_size = int(np.ceil(Config.T_VERT*Config.T_HOR*(1 - (kt_dist_accum/n_views))))
            # print(f'{video},{segment},{kt_dist_accum/n_views},{vp_size}')
            for t_id in new_rank[:vp_size]:
                if self.prefetch_mode == 'auto':
                    quality = qualities[t_id - 2]
                    q_index = int(1.5*quality**2 - 0.5*quality)
                elif self.prefetch_mode == 'hq':
                    q_index = Config.SUPPORTED_QUALITIES[-1]
                else:
                    q_index = Config.SUPPORTED_QUALITIES[0]
                #for q_index in Config.SUPPORTED_QUALITIES:
                Thread(
                    target=self.buffer_tile,
                    args=(t_id - 1, segment, video, q_index),
                    daemon=True
                ).start()
            # There is a ranking for (video, segment) and it was succesfully updated 
            return True
        # There is no ranking for (video, segment) yet
        return False

    def buffer_tile_into_init_buffer(self, tile, segment, video, quality):
        filename = f'seg_dash_track{tile}_{segment}.m4s'
        key = f'{video}:{segment}:{tile}:{quality}'
        if not self.init_buffer.exists(key):
            return self.init_buffer.set(key, self.fetch(Config.T_HOR, Config.T_VERT, video, quality, filename))
    
    def buffer_tile(self, tile, segment, video, quality, user=-1, save_to_collective_buffer=True):
        # print(f"[buffer_tile] seg_dash_track{tile}_{segment}.m4s")
        filename = f'seg_dash_track{tile}_{segment}.m4s'
        key = f'{video}:{segment}:{tile}:{quality}'
        # Redis.setnx (set if not exists) could've been used next but 
        # then the fetch call would always be made which is precisely
        # what we are trying to avoid
        if not self.collective_buffer.exists(key):
            if user < 0:
                return self.collective_buffer.set(key, self.fetch(Config.T_HOR, Config.T_VERT, video, quality, filename))
            else:
                user_key = f'{user}:{key}'
                tile_bytes = self.fetch(Config.T_HOR, Config.T_VERT, video, quality, filename)
                self.user_buffer.set(user_key, tile_bytes, ex=2) # Set 2 sec. expire time to ephemeral user buffer
                if save_to_collective_buffer:
                    self.collective_buffer.set(key, tile_bytes)
                return

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

    def remove_segment_from_collective_buffer(self, key):
        keys = self.collective_buffer.keys(f'{key}:*')
        self.collective_buffer.delete(*keys)
        del self.segment_rankings[tuple(map(int, key.split(':')))]
    
    def remove_segment_from_user_buffer(self, key):
        keys = self.user_buffer.keys(f'{key}:*')
        if keys:
            self.user_buffer.delete(*keys)

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
            u_id = p_data['u_id']
            qualities = p_data['qualities']
            order = p_data['order']
            if not p.init_buffer_keys.contains(f'{v_id}:{s_id}'): #(int(os.getenv("VIEWPORT_SIZE")) > 1):
                Thread(
                    target=p.prefetch_segment,
                    args=(v_id, s_id, u_id, qualities, order),
                    daemon=True
                ).start()
            # if p.buffer_keys.contains(f'{v_id}:{s_id}'):
            #     Thread(
            #         target=p.update_segment,
            #         args=(v_id, s_id, order),
            #         daemon=True
            #     ).start()



if __name__ == "__main__":
    main()