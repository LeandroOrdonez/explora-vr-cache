import os, re, pickle, sys
import pandas as pd
import numpy as np
# from multiprocessing import Process, Manager, Queue
from threading import Thread
from queue import Queue
from instance.config import Config

sys.path.append(r'./instance/')
class PrefetchBufferHandler:

    def __init__(self, fn):
        print(f'__init__')
        # self.manager = Manager()
        self.fn = fn
        self.buffer = dict()
        self.init_buffer = dict() # It buffers the first segments of all videos in the catalog and keeps them in memory
        self.buffer_keys = Queue()
        self.buffer_keys_len = int(os.getenv("BUFFER_SIZE"))
        self.n_queries = 0
        self.n_hits = 0
        if os.getenv('PERFECT_PREDICTION') == 'true':
            for video in Config.VIDEO_CATALOG:
                args = (Config.T_HOR, Config.T_VERT, video, Config.SUPPORTED_QUALITIES[-1], 'seg_dash_trackX_X.m4s') # Only important elements from this tuple at initialization are T_VER, T_HOR and the video ID, the rest are placeholders, basically 
                Thread(
                    target=self.prefetch_segment_into_init_buffer,
                    args=(args, int(video), 0, 0),
                ).start()
            self.user_traces = self.load_user_traces()
        else:
            self.prefetch_models = {
                'v0': pickle.load(open(f'./instance/model_files/em_v0_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
                'v2': pickle.load(open(f'./instance/model_files/em_v2_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
                'v4': pickle.load(open(f'./instance/model_files/em_v4_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb'))
            }

    def __call__(self, *args):
        # print(f'[Prefetch_Handler] __call__ with args: {args}')
        self.n_queries += 1
        # print(f'[Prefetch_Handler] Current buffer: {self.buffer.keys()}')
        if(os.getenv('ENABLE_PREFETCHING') == 'false'):
            return self.fn(*args)
        video, quality, tile, seg, user_id = self.get_video_segment_and_tile(args)
        # resp = None
        Thread(
            target=self.run_prefetch,
            args=(video, tile, seg, user_id, args)
        ).start()
        if ((video, seg) in self.init_buffer) and ((tile, quality) in self.init_buffer[(video, seg)]):
            # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
            self.n_hits += 1
            print(f'[Prefetch_Handler] hit-ratio: {self.n_hits}/{self.n_queries} = {round((self.n_hits/self.n_queries)*100, 2)}%')
            return self.init_buffer[(video, seg)][(tile, quality)]
        elif ((video, seg) in self.buffer) and ((tile, quality) in self.buffer[(video, seg)]):
            # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
            self.n_hits += 1
            print(f'[Prefetch_Handler] hit-ratio: {self.n_hits}/{self.n_queries} = {round((self.n_hits/self.n_queries)*100, 2)}%')
            return self.buffer[(video, seg)][(tile, quality)]
        else:
            return self.fn(*args)
            
    def run_prefetch(self, video, tile, segment, user_id, args):
        # Prefetch current segment
        if (video, segment) not in self.buffer and (int(os.getenv("BUFFER_SEQ_LENGTH")) > 1):
            Thread(
                target=self.prefetch_segment,
                args=(args, video, segment, tile, False, user_id),
            ).start()
        # Prefetch next segment
        if (video, segment + 1) not in self.buffer:
            Thread(
                target=self.prefetch_segment,
                args=(args, video, segment, tile, True, user_id),
            ).start()

    def get_video_segment_and_tile(self, args):
        filename = args[4]
        video = args[2]
        quality = args[3]
        user_id = args[5]
        tile, seg = re.findall(r'(\d+)\_(\d+).', filename)[0]
        return int(video), int(quality), int(tile), int(seg), user_id
    
    def prefetch_segment_into_init_buffer(self, args, video, segment, tile):
        self.init_buffer[(video, segment + 1)] = dict()
        for i_tile in range(Config.T_VERT*Config.T_HOR):
            for q_index in Config.SUPPORTED_QUALITIES:
                # print(f"PREFETCHING TILE {i_tile + 1} Q{q_index} FROM SEGMENT {actual_segment + 1} VIDEO {video}")
                Thread(
                    target=self.fetch_tile,
                    args=(i_tile, segment, video, q_index, args, True)
                ).start()


    def prefetch_segment(self, args, video, segment, tile, next_segment=False, user_id=None):
        # Tiles and Segments in the predictive model are zero-indexed
        actual_segment = segment - 1 if not next_segment else segment
        if self.buffer_keys.qsize() >= self.buffer_keys_len:
            first_key = self.buffer_keys.get()
            del self.buffer[first_key]
        # self.buffer[(video, actual_segment)] = self.manager.dict()
        self.buffer[(video, actual_segment + 1)] = dict()
        self.buffer_keys.put((video, actual_segment + 1))
        if os.getenv('PERFECT_PREDICTION') == 'true':
            if user_id:
                # print("TODO: prefetch for specific users with perfect prediction")
                vp_tiles = np.array(self.user_traces[video][user_id][actual_segment + 1]) - 1 # offset for zero-indexing 
                pred_tiles = [(t, Config.SUPPORTED_QUALITIES[-1] if t in vp_tiles else Config.SUPPORTED_QUALITIES[0]) for t in range(Config.T_HOR*Config.T_VERT)]
                for (i_tile, q_index) in pred_tiles:
                    Thread(
                        target=self.fetch_tile,
                        args=(i_tile, actual_segment, video, q_index, args)
                    ).start()
            else:
                for i_tile in range(Config.T_VERT*Config.T_HOR):
                    for q_index in Config.SUPPORTED_QUALITIES:
                        # print(f"PREFETCHING TILE {i_tile + 1} Q{q_index} FROM SEGMENT {actual_segment + 1} VIDEO {video}")
                        Thread(
                            target=self.fetch_tile,
                            args=(i_tile, actual_segment, video, q_index, args)
                        ).start()
        else:
            pred_seg = self.prefetch_models[f'v{video}'].predict_next_segment(actual_segment - 1, tile - 1) if next_segment else self.prefetch_models[f'v{video}'].predict_current_segment(actual_segment)
            # print(pred_seg)
            for i_tile in pred_seg:
                Thread(
                    target=self.fetch_tile,
                    args=(i_tile, actual_segment, video, Config.SUPPORTED_QUALITIES[-1], args)
                ).start()
                
            # print(f'[prefetch] Current buffer: {[(k, v.items()) for k, v in self.buffer.items()]}')

    def fetch_tile(self, tile, segment, video, quality, args, into_init_buffer=False):
        tmp_args = list(args)
        tmp_args[2] = video
        tmp_args[3] = quality
        tmp_args[4] = f'seg_dash_track{tile + 1}_{segment + 1}.m4s'
        if into_init_buffer:
            self.init_buffer[(video, segment + 1)][(tile + 1, quality)] = self.fn(*tuple(tmp_args))
        else:
            self.buffer[(video, segment + 1)][(tile + 1, quality)] = self.fn(*tuple(tmp_args))

    def load_user_traces(self):
        user_traces = {}
        for video in Config.VIDEO_CATALOG:
            user_traces[int(video)] = {}
            for user in range (1, len(os.listdir(f'{Config.USER_TRACES_PATH}/{video}'))+1):
                user_traces[int(video)][user] = {}
                user_df = pd.read_csv(f'{Config.USER_TRACES_PATH}/{video}/queries_u{user}.txt', names=['segment', 'tile', 'viewport', 'url'])
                for segment, df in user_df.groupby('segment'):
                    vp_tiles = df[df['viewport'] == True]['tile'].tolist()
                    user_traces[int(video)][user][segment] = vp_tiles
        return user_traces

