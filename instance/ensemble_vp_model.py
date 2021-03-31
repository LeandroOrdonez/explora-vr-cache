import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from viewport_prediction_model import ViewportPredictionModel
from transition_model import TransitionModel

class EnsembleVPModel:
    def __init__(self, threshold=90, popularity_df=None, t_vert=4, t_hor=4, n_segments=50):
        self.tm = TransitionModel(t_vert=t_vert, t_hor=t_hor, n_segments=n_segments)
        self.vpm = ViewportPredictionModel(threshold=threshold, t_vert=t_vert, t_hor=t_hor, n_segments=n_segments, popularity_df=popularity_df)
        
    def fit(self, X, y):
        # Fixation sequence part
        self.vpm.fit()
        # Transition Model part
        self.tm.fit(X, y)
        
        return self.vpm.viewport_sequence
    
    def predict_current_segment(self, segment):
        assert segment < self.vpm.n_segments, f'segment {segment} out of range! (n_segments={self.vpm.n_segments})'
        vpm_seq = list(self.vpm.viewport_sequence.loc[segment].dropna().astype(int))
        return vpm_seq
    
    def predict_next_segment(self, segment, tile):
        assert segment < self.vpm.n_segments-1, f'segment {segment} out of range! (n_segments={self.vpm.n_segments})'
        tm_tile = self.tm.predict(segment, tile)
        vpm_seq = self.predict_current_segment(segment+1)
        if tm_tile in vpm_seq: # predicted tile from transition model is already in the predicted fixation sequence  
            vpm_seq.insert(0, vpm_seq.pop(vpm_seq.index(tm_tile)))
            pred_set = vpm_seq
        else: # get rid of the last tile from the predicted fixation sequence and append the tile predicted by the transition model
            pred_set = [int(tm_tile)] + vpm_seq[:-1] if not pd.isna(tm_tile) else vpm_seq
        return pred_set
        
        