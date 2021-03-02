import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class FixationSequencesModel:
    def __init__(self, k=3, popularity_df=None, t_vert=4, t_hor=4, n_segments=50):
        self.k = k
        self.popularity_df = popularity_df
        self.t_vert = 4
        self.t_hor = 4
        self.n_segments = n_segments
        self.fixation_sequence = pd.DataFrame()
        
    def fit(self):
        assert not self.popularity_df.empty, f'The popularity data frame cannot be empty'
        for seg, df in self.popularity_df.groupby('segment'):
            tmp_df = df.sort_values(by='sample_proportion', ascending=False).head(self.k).copy().reset_index(drop=True).reset_index()
            tmp_df = tmp_df.rename(columns={'index': 'k'})
            tmp_df['k'] = tmp_df['k'] + 1
            self.fixation_sequence = self.fixation_sequence.append(tmp_df)
        self.fixation_sequence = self.fixation_sequence.pivot(index='segment', columns='k', values='tile')
        return self.fixation_sequence
    
    def predict(self, segment, k=1):
#         assert segment < self.n_segments, f'segment {segment} out of range! (n_segments={n_segments})'
        return self.fixation_sequence[k][segment] if segment < self.n_segments else np.nan
        
        