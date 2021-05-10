import os, re, pickle, sys, json
import pandas as pd
import numpy as np
# from multiprocessing import Process, Manager, Queue
from threading import Thread
from queue import Queue
from instance.config import Config
from util.redis_queue import RedisQueue
from redis import Redis

sys.path.append(r'./instance/')
class PrefetchBufferHandler:

    def __init__(self, fn):
        print(f'__init__')
        # self.manager = Manager()
        self.fn = fn
        self.buffer = Redis(host=os.getenv("REDIS_HOST"), db=0)
        self.init_buffer = Redis(host=os.getenv("REDIS_HOST"), db=1) # It buffers the first segments of all videos in the catalog and keeps them in memory
        self.counters = Redis(host=os.getenv("REDIS_HOST"), db=2)
        self.publisher = Redis(host=os.getenv("REDIS_HOST"), charset="utf-8", decode_responses=True)


    def __call__(self, *args):
        # print(f'[Prefetch_Handler] __call__ with args: {args}')
        n_queries = self.counters.incr('n_queries')
        # print(f'[Prefetch_Handler] Current buffer: {self.buffer.keys()}')
        if(os.getenv('ENABLE_PREFETCHING') == 'false'):
            return self.fn(*args)
        video, quality, tile, seg, ct = self.get_video_segment_and_tile(args)
        # if ct:
        #     self.publisher.publish("prefetch", json.dumps(args))
        tile_key = f'{video}:{seg}:{tile}:{quality}'
        hq_tile_key = f'{video}:{seg}:{tile}:{Config.SUPPORTED_QUALITIES[-1]}'
        if self.init_buffer.exists(tile_key):
            # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
            n_hits = self.counters.incr('n_hits')
            print(f'[Prefetch_Handler] hit-ratio: {n_hits}/{n_queries} = {round((n_hits/n_queries)*100, 2)}%')
            quality_upgrade = False
            return quality_upgrade, self.init_buffer.get(tile_key)
        elif self.buffer.exists(tile_key):
            # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
            n_hits = self.counters.incr('n_hits')
            print(f'[Prefetch_Handler] hit-ratio: {n_hits}/{n_queries} = {round((n_hits/n_queries)*100, 2)}%')
            quality_upgrade = False
            return quality_upgrade, self.buffer.get(tile_key)
        # # If LQ tile requested, but HQ version in buffer then return HQ tile
        # elif quality == Config.SUPPORTED_QUALITIES[0] and self.buffer.exists(hq_tile_key): # If LQ tile requested, but HQ version in buffer then return HQ
        #      # resp = self.buffer[(video, seg)][tile] if tile in self.buffer[(video, seg)] else self.fn(*args)
        #     n_hits = self.counters.incr('n_hits')
        #     print(f'[Prefetch_Handler] hit-ratio: {n_hits}/{n_queries} = {round((n_hits/n_queries)*100, 2)}%')
        #     quality_upgrade = True
        #     return quality_upgrade, self.buffer.get(hq_tile_key)
        else:
            return self.fn(*args)

    def get_video_segment_and_tile(self, args):
        filename = args[4]
        video = args[2]
        quality = args[3]
        tile, seg = re.findall(r'(\d+)\_(\d+).', filename)[0]
        ct = args[8]
        return int(video), int(quality), int(tile), int(seg), bool(ct)
