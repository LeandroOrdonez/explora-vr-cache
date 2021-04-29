import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
sys.path.append(r'./instance/')
from config import Config

class SegmentRank:
    
    def __init__(self):
        self.n_views = 0
        self.kt_dist_accum = 0.0
        self.agg_rank = pd.DataFrame(index=np.arange(2, Config.T_VERT*Config.T_HOR + 2), columns=['sum'], data=np.zeros((Config.T_VERT*Config.T_HOR)))

    def update_rank(self, seg_rank):
        self.n_views += 1
        s_df = pd.DataFrame(index=seg_rank, columns=['sum'], data=[t[0]+1 for t in enumerate(seg_rank)])
    #     consolidate_rank['sum'] = consolidate_rank['sum'] + s_df['sum']
        self.agg_rank['sum'] = self.agg_rank['sum'] + s_df['sum']
        self.agg_rank['avg'] = self.agg_rank['sum'] / self.n_views
        curr_rank = np.array(self.agg_rank.sort_values(by='avg').index)
    #     print(f'Current segment:     {seg_rank}')
    #     print(f'Ongoing consolidate: {curr_rank}\n')
        self.kt_dist_accum += (kendalltau(curr_rank, np.array(seg_rank)).correlation + 1) / 2
        return curr_rank, self.kt_dist_accum, self.n_views