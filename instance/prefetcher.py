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
        for video in Config.VIDEO_CATALOG:
            args = (Config.T_HOR, Config.T_VERT, video, Config.SUPPORTED_QUALITIES[-1], 'seg_dash_trackX_X.m4s') # Only important elements from this tuple at initialization are T_VER, T_HOR and the video ID, the rest are placeholders, basically 
            Thread(
                target=self.prefetch_segment_into_init_buffer,
                args=(args, int(video), 0, 0),
                daemon=True
            ).start()
        if os.getenv('PERFECT_PREDICTION') == 'true':
            self.user_traces = self.load_user_traces()
        else:
            self.prefetch_models = {}
            for v in Config.VIDEO_CATALOG:	
	                self.prefetch_models[f'v{v}'] = {}	
	                for fold in range(1,9):	
	                    self.prefetch_models[f'v{v}'][f'f{fold}'] = pickle.load(open(f'./instance/model_files/fold_{fold}/em_v{v}_k16.pkl','rb'))

        print(f'[Prefetcher] OK, Listening...')


    def run_prefetch(self, video, tile, segment, vp_size, user_id, fold, args):
        # Prefetch current segment
        # print(f"[run_prefetch] {args}")
        if not (self.buffer_keys.contains(f'{video}:{segment}') or self.init_buffer_keys.contains(f'{video}:{segment}')): #(int(os.getenv("VIEWPORT_SIZE")) > 1):
            Thread(
                target=self.prefetch_segment,
                args=(args, video, segment, tile, vp_size, False, user_id, fold),
                daemon=True
            ).start()
        # Prefetch next segment
        if not self.buffer_keys.contains(f'{video}:{segment + 1}'):
            Thread(
                target=self.prefetch_segment,
                args=(args, video, segment, tile, vp_size, True, user_id, fold),
                daemon=True
            ).start()
    
    def prefetch_segment_into_init_buffer(self, args, video, segment, tile):
        # self.init_buffer[(video, segment + 1)] = dict()
        self.init_buffer_keys.put(f'{video}:{segment + 1}')
        for i_tile in range(Config.T_VERT*Config.T_HOR):
            for q_index in Config.SUPPORTED_QUALITIES:
                # print(f"PREFETCHING TILE {i_tile + 1} Q{q_index} FROM SEGMENT {actual_segment + 1} VIDEO {video}")
                Thread(
                    target=self.fetch_tile,
                    args=(i_tile, segment, video, q_index, args, True),
                    daemon=True
                ).start()

    def prefetch_segment(self, args, video, segment, tile, vp_size, next_segment=False, user_id=None, fold=1):
        # print(f"[prefetch_segment] {args}")
        # Tiles and Segments in the predictive model are zero-indexed
        actual_segment = segment - 1 if not next_segment else segment
        if self.buffer_keys.qsize() >= self.buffer_keys_len:
            first_key = self.buffer_keys.get().decode('utf-8')
            self.remove_segment_from_buffer(first_key)
        # self.buffer[(video, actual_segment + 1)] = dict()
        self.buffer_keys.put(f'{video}:{actual_segment + 1}')
        if os.getenv('PERFECT_PREDICTION') == 'true':
            if user_id:
                vp_tiles = np.array(self.user_traces[video][user_id][actual_segment + 1]) - 1 # offset for zero-indexing 
                pred_tiles = [(t, Config.SUPPORTED_QUALITIES[-1] if t in vp_tiles else Config.SUPPORTED_QUALITIES[0]) for t in range(Config.T_HOR*Config.T_VERT)]
                for (i_tile, q_index) in pred_tiles:
                    Thread(
                        target=self.fetch_tile,
                        args=(i_tile, actual_segment, video, q_index, args),
                        daemon=True
                    ).start()
            else:
                for i_tile in range(Config.T_VERT*Config.T_HOR):
                    for q_index in Config.SUPPORTED_QUALITIES:
                        # print(f"PREFETCHING TILE {i_tile + 1} Q{q_index} FROM SEGMENT {actual_segment + 1} VIDEO {video}")
                        Thread(
                            target=self.fetch_tile,
                            args=(i_tile, actual_segment, video, q_index, args),
                            daemon=True
                        ).start()
        else:
            # Predict VP tiles (HQ tiles)
            pred_seg = self.prefetch_models[f'v{video}'][f'f{fold}'].predict_next_segment(actual_segment - 1, tile - 1) if next_segment else self.prefetch_models[f'v{video}'][f'f{fold}'].predict_current_segment(actual_segment)
            if vp_size > 0:
                pred_seg = pred_seg[:vp_size]
            pred_tiles = [(t, Config.SUPPORTED_QUALITIES[-1] if t in pred_seg else Config.SUPPORTED_QUALITIES[0]) for t in range(Config.T_HOR*Config.T_VERT)]
            # Prefetch LQ tiles only:
            # pred_tiles = [(t, Config.SUPPORTED_QUALITIES[0]) for t in range(Config.T_HOR*Config.T_VERT)]
            for (i_tile, q_index) in pred_tiles:
                Thread(
                    target=self.fetch_tile,
                    args=(i_tile, actual_segment, video, q_index, args),
                    daemon=True
                ).start()
                
            # print(f'[prefetch] Current buffer: {[(k, v.items()) for k, v in self.buffer.items()]}')

    def fetch_tile(self, tile, segment, video, quality, args, into_init_buffer=False):
        # print(f"[fetch_tile] seg_dash_track{tile + 1}_{segment + 1}.m4s")
        tmp_args = list(args)
        tmp_args[2] = video
        tmp_args[3] = quality
        tmp_args[4] = f'seg_dash_track{tile + 1}_{segment + 1}.m4s'
        if into_init_buffer:
            return self.init_buffer.set(f'{video}:{segment+ 1}:{tile + 1}:{quality}', self.fetch(*tuple(tmp_args)))
        else:
            return self.buffer.set(f'{video}:{segment+ 1}:{tile + 1}:{quality}', self.fetch(*tuple(tmp_args)))

    def fetch(self, t_hor, t_vert, video_id, quality, filename, vp_size=-1, user_id=None, fold=1, ct=False):
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

    def load_user_traces(self):
        user_traces = {}
        for video in Config.VIDEO_CATALOG:
            user_traces[int(video)] = {}
            for user in range (1, len(os.listdir(f'{Config.USER_TRACES_PATH}/{video}'))+1):
                user_traces[int(video)][user] = {}
                user_df = pd.read_csv(f'{Config.USER_TRACES_PATH}/{video}/queries_u{user}.txt', names=['segment', 'tile', 'viewport', 'url'])
                for segment, df in user_df.groupby('segment'):
                    vp_tiles = df[df['viewport'] == True]['tile'].tolist() if int(os.getenv("VIEWPORT_SIZE")) < 0 else df['tile'].tolist()[:int(os.getenv("VIEWPORT_SIZE"))]
                    user_traces[int(video)][user][segment] = vp_tiles
        return user_traces

    def remove_segment_from_buffer(self, key):
        keys = self.buffer.keys(f'{key}:*')
        self.buffer.delete(*keys)

    def get_video_segment_and_tile(self, args):
        filename = args[4]
        video = args[2]
        quality = args[3]
        vp_size = args[5]
        user_id = args[6]
        fold = args[7]
        ct = args[8]
        tile, seg = re.findall(r'(\d+)\_(\d+).', filename)[0]
        return int(video), int(quality), int(tile), int(seg), int(vp_size), user_id, int(fold), bool(ct)

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
            args = json.loads(message.get("data"))
            print(f"[Prefetcher] received: {args}")
            video, quality, tile, seg, vp_size, user_id, fold, ct = p.get_video_segment_and_tile(args)
            # # run_prefetch call inside a new Thread. Direct call doesn't work.
            # Thread(
            #     target=p.run_prefetch,
            #     args=(video, tile, seg, vp_size, user_id, fold, args),
            #     daemon=True
            # ).start()
            
            # Prefetch current segment
            if not (p.buffer_keys.contains(f'{video}:{seg}') or p.init_buffer_keys.contains(f'{video}:{seg}')): #(int(os.getenv("VIEWPORT_SIZE")) > 1):
                Thread(
                    target=p.prefetch_segment,
                    args=(args, video, seg, tile, vp_size, False, user_id, fold),
                    daemon=True
                ).start()
            # Prefetch next segment
            if not p.buffer_keys.contains(f'{video}:{seg + 1}'):
                Thread(
                    target=p.prefetch_segment,
                    args=(args, video, seg, tile, vp_size, True, user_id, fold),
                    daemon=True
                ).start()


if __name__ == "__main__":
    main()