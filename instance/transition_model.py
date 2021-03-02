import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class TransitionModel:
    def __init__(self, supp_mats=list(), conf_mats=list(), transition_probs=list(), t_vert=4, t_hor=4, n_segments=50):
        self.supp_mats = supp_mats
        self.conf_mats = conf_mats
        self.transition_probs = transition_probs
        self.t_vert = 4
        self.t_hor = 4
        self.n_segments = n_segments
        
    def fit(self, X, y):
        X['tile'] = y
        self.supp_mats = list()
        self.conf_mats = list()
        self.transition_probs = list()
        for seg in range(self.n_segments-1):
            curr_seg = X[X['segment'] == seg].copy()
            next_seg = X[X['segment'] == (seg + 1)].copy()
            trans_matrix = curr_seg.set_index('user').join(next_seg.set_index('user'), lsuffix='_curr', rsuffix='_next').groupby(['tile_curr', 'tile_next']).count().reset_index().pivot('tile_curr', 'tile_next', 'segment_curr').fillna(0)
            for ri in set(range(self.t_hor*self.t_vert)) - set(trans_matrix.index):
#                 s = pd.Series(np.zeros(trans_matrix.shape[1]))
#                 s.name = ri
#                 trans_matrix = trans_matrix.append(s)
                trans_matrix.loc[ri] = pd.Series(np.zeros(trans_matrix.shape[1]))
            for ci in set(range(self.t_hor*self.t_vert)) - set(trans_matrix.columns):
                trans_matrix[ci] = pd.Series(np.zeros(trans_matrix.shape[0]))
            trans_matrix = trans_matrix.sort_index().sort_index(axis=1).fillna(0)
            trans_supp_df = (trans_matrix.T/trans_matrix.sum().sum()).T
            trans_conf_df = (trans_matrix.T/trans_matrix.T.sum()).T
            self.supp_mats.append(trans_supp_df)
            self.conf_mats.append(trans_conf_df)
            self.transition_probs.append(1 - np.diag(trans_matrix).sum()/trans_matrix.sum().sum())
    
    def predict_proba(self, segment, tile):
        if segment >= self.n_segments - 1:
            return ([np.nan], [np.nan])
        top_2 = self.conf_mats[int(segment)].iloc[int(tile),:].nlargest(2)
        if not top_2.empty:
            if all(top_2.duplicated(keep=False)): # If the two top tiles have the same probability of occuring
                if self.transition_probs[int(segment)] < 0.5: 
                    return ([tile], [1 - self.transition_probs[int(segment)]])
                else:
                    return ([int(top_2.index[0])], [top_2.iloc[0]]) if top_2.index[0] != tile else ([int(top_2.index[1])], [top_2.iloc[1]])
            else:
                return ([int(top_2.index[0])], [top_2.iloc[0]])
        elif self.transition_probs[int(segment)] >= 0.5: # High probability of moving to a different tile
            return ([np.nan], [np.nan]) # Then is not possible to predict that the user is going to stay looking at the same tile
        else:
            return ([tile], [1 - self.transition_probs[int(segment)]])

    def predict(self, segment, tile):
        return self.predict_proba(segment, tile)[0][0]
        
        