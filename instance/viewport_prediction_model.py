import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class ViewportPredictionModel:
    def __init__(self, threshold=90, popularity_df=None, t_vert=4, t_hor=4, n_segments=50):
        self.threshold=threshold
        self.popularity_df = popularity_df
        self.t_vert = 4
        self.t_hor = 4
        self.n_segments = n_segments
        self.viewport_sequence = pd.DataFrame()
        
    def fit(self):
        assert not self.popularity_df.empty, f'The popularity data frame cannot be empty'
        for seg, df in self.popularity_df.groupby('segment'):
            tmp_df = df.loc[df['sample_proportion'].sort_values(ascending=False).cumsum()<=(self.threshold/100)].copy()
            tmp_df = tmp_df.sort_values(by='sample_proportion', ascending=False).reset_index(drop=True).reset_index()
            tmp_df = tmp_df.rename(columns={'index': 'k'})
            tmp_df['k'] = tmp_df['k'] + 1
            self.viewport_sequence = self.viewport_sequence.append(tmp_df)
        self.viewport_sequence = self.viewport_sequence.pivot(index='segment', columns='k', values='tile')
        return self.viewport_sequence
    
    def predict(self, segment, k=1):
#         assert segment < self.n_segments, f'segment {segment} out of range! (n_segments={n_segments})'
        return self.viewport_sequence[k][segment] if segment < self.n_segments else np.nan
        
        