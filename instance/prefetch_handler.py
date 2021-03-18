import os, re, pickle, sys
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
        self.buffer_keys = Queue()
        self.buffer_keys_len = int(os.getenv("BUFFER_SIZE"))
        self.prefetch_models = {
            'v0': pickle.load(open(f'./instance/model_files/em_v0_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
            'v2': pickle.load(open(f'./instance/model_files/em_v2_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
            'v4': pickle.load(open(f'./instance/model_files/em_v4_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb'))
        }
        self.n_queries = 0
        self.n_hits = 0
        if os.getenv('PERFECT_PREDICTION') == 'true':
            for video in Config.VIDEO_CATALOG:
                args = (Config.T_HOR, Config.T_VERT, video, Config.SUPPORTED_QUALITIES[-1], 'seg_dash_trackX_X.m4s') # Only important elements from this tuple at initialization are T_VER, T_HOR and the video ID, the rest are placeholders, basically 
                Thread(
                    target=self.prefetch,
                    args=(args, video, 0, 0, True),
                ).start()

    def __call__(self, *args):
        # print(f'[Prefetch_Handler] __call__ with args: {args}')
        self.n_queries += 1
        print(f'[Prefetch_Handler] hit-ratio: {self.n_hits}/{self.n_queries} = {round((self.n_hits/self.n_queries)*100, 2)}%')
        print(f'[Prefetch_Handler] Current buffer: {self.buffer.items()}')
        if(os.getenv('ENABLE_PREFETCHING') == 'false'):
            return self.fn(*args)
        video, quality, tile, seg = self.get_video_segment_and_tile(args)
        resp = None
        if (video, seg) in self.buffer:
            # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
            if tile in self.buffer[(video, seg)]:
                self.n_hits += 1 
                resp = self.buffer[(video, seg)][tile]
            else:
                resp = self.fn(*args)
        else:
            resp = self.fn(*args)
            if int(os.getenv("BUFFER_SEQ_LENGTH")) > 1:
                Thread(
                    target=self.prefetch,
                    args=(args, video, seg, tile),
                ).start()
                # self.prefetch(args, video, seg, tile)
        if (video, seg+1) not in self.buffer:
            Thread(
                target=self.prefetch,
                args=(args, video, seg, tile, True),
            ).start()
            # self.prefetch(args, video, seg, tile, next_segment=True)
        return resp

    def get_video_segment_and_tile(self, args):
        filename = args[4]
        video = args[2]
        quality = args[3]
        tile, seg = re.findall(r'(\d+)\_(\d+).', filename)[0]
        return int(video), int(quality), int(tile), int(seg)
    
    def prefetch(self, args, video, segment, tile, next_segment=False, user_id=None):
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
                print("TODO: prefetch for specific users with perfect prediction")
                pass
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

    def fetch_tile(self, tile, segment, video, quality, args):
        tmp_args = list(args)
        tmp_args[2] = video
        tmp_args[3] = quality
        tmp_args[4] = f'seg_dash_track{tile + 1}_{segment + 1}.m4s'
        self.buffer[(video, segment + 1)][(tile + 1, quality)] = self.fn(*tuple(tmp_args))
