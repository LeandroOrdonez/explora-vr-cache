import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from fixation_sequences_model import FixationSequencesModel
from transition_model import TransitionModel

class EnsembleModel:
    def __init__(self, k=4, popularity_df=None, t_vert=4, t_hor=4, n_segments=50):
        self.tm = TransitionModel(t_vert=t_vert, t_hor=t_hor, n_segments=n_segments)
        self.fsm = FixationSequencesModel(k=k, t_vert=t_vert, t_hor=t_hor, n_segments=n_segments, popularity_df=popularity_df)
        
    def fit(self, X, y):
        # Fixation sequence part
        self.fsm.fit()
        # Transition Model part
        self.tm.fit(X, y)
        
        return self.fsm.fixation_sequence
    
    def predict_current_segment(self, segment):
        assert segment < self.fsm.n_segments, f'segment {segment} out of range! (n_segments={self.fsm.n_segments})'
        fsm_seq = list(self.fsm.fixation_sequence.loc[segment])
        return fsm_seq
    
    def predict_next_segment(self, segment, tile):
        assert segment < self.fsm.n_segments-1, f'segment {segment} out of range! (n_segments={self.fsm.n_segments})'
        tm_tile = self.tm.predict(segment, tile)
        fsm_seq = self.predict_current_segment(segment+1)
        if tm_tile in fsm_seq: # predicted tile from transition model is already in the predicted fixation sequence  
            fsm_seq.insert(0, fsm_seq.pop(fsm_seq.index(tm_tile)))
            pred_set = fsm_seq
        else: # get rid of the last tile from the predicted fixation sequence and append the tile predicted by the transition model
            pred_set = [int(tm_tile)] + fsm_seq[:-1] if not pd.isna(tm_tile) else fsm_seq
        return pred_set
        
        