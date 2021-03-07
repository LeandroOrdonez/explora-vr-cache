import os, re, pickle, sys
# from multiprocessing import Process, Manager, Queue
from threading import Thread
from queue import Queue

sys.path.append(r'./instance/')
class PrefetchBufferHandler:

    def __init__(self, fn):
        print(f'__init__')
        # self.manager = Manager()
        self.fn = fn
        self.buffer = dict()
        self.buffer_keys = Queue()
        self.buffer_keys_len = int(os.getenv("BUFFER_SIZE")) / int(os.getenv("BUFFER_SEQ_LENGTH"))
        self.prefetch_models = {
            'v0': pickle.load(open(f'./instance/model_files/em_v0_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
            'v2': pickle.load(open(f'./instance/model_files/em_v2_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb')),
            'v4': pickle.load(open(f'./instance/model_files/em_v4_k{os.getenv("BUFFER_SEQ_LENGTH")}.pkl','rb'))
        }
        self.n_queries = 0
        self.n_hits = 0

    def __call__(self, *args):
        # print(f'[Prefetch_Handler] __call__ with args: {args}')
        self.n_queries += 1
        print(f'[Prefetch_Handler] hit-ratio: {self.n_hits}/{self.n_queries} = {round((self.n_hits/self.n_queries)*100, 2)}%')
        # print(f'[Prefetch_Handler] Current buffer: {self.buffer.items()}')
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
        filename = args[6]
        video = args[4]
        quality = args[5]
        tile, seg = re.findall(r'(\d+)\_(\d+).', filename)[0]
        return int(video), int(quality), int(tile), int(seg)
    
    def prefetch(self, args, video, segment, tile, next_segment=False):
        actual_segment = segment - 1 if not next_segment else segment
        if self.buffer_keys.qsize() >= self.buffer_keys_len:
            first_key = self.buffer_keys.get()
            del self.buffer[first_key]
        # self.buffer[(video, actual_segment)] = self.manager.dict()
        self.buffer[(video, actual_segment + 1)] = dict()
        self.buffer_keys.put((video, actual_segment + 1))
        pred_seg = self.prefetch_models[f'v{video}'].predict_next_segment(actual_segment - 1, tile - 1) if next_segment else self.prefetch_models[f'v{video}'].predict_current_segment(actual_segment)
        # print(pred_seg)
        for i_tile in pred_seg:
            pred_args = list(args)
            pred_args[4] = video
            pred_args[6] = f'seg_dash_track{i_tile + 1}_{actual_segment + 1}.m4s'
            self.buffer[(video, actual_segment + 1)][i_tile + 1] = self.fn(*tuple(pred_args))
        # print(f'[prefetch] Current buffer: {[(k, v.items()) for k, v in self.buffer.items()]}')


