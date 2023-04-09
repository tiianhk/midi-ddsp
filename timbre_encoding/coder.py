import os
import tensorflow as tf
import librosa
import numpy as np
import openl3
import joblib
import warnings
warnings.filterwarnings("ignore")


class TimbreCoder():
    """class for mapping audio into timbre embeddings."""
    
    def __init__(self, method):
        self.method = method
        self.model_dir = os.path.join('./timbre_encoding/models/', method)
        self.preprocessor = self._load_preprocessor()
        self.coder = self._load_coder()
        self.sample_rate = 16000
    
    def _load_preprocessor(self):
        if self.method == 'lda':
            """time averaged mfcc"""
            return lambda y: np.mean(librosa.feature.mfcc(y=y, sr=self.sample_rate), axis=-1)
        elif self.method == 'openl3':
            return lambda y: y
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """melspectrogram"""
            return lambda y: librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, power=1)[...,np.newaxis]
        else:
            raise ValueError('invalid method')
    
    def _load_coder(self):
        if self.method == 'lda':
            """pretrained standardization and LDA transformation"""
            model = joblib.load(self.model_dir+'.joblib')
            return lambda x: np.squeeze(model.transform(x.reshape(1, -1)))
        elif self.method == 'openl3':
            """time averaged openl3 embedding"""
            model = openl3.models.load_audio_embedding_model(
                input_repr="mel128", content_type="music", embedding_size=512)
            return lambda x: np.mean(openl3.get_audio_embedding(x, sr=self.sample_rate,
                model=model, center=False, hop_size=0.5, verbose=0)[0], axis=0)
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """pretrained triplet models"""
            model = tf.keras.models.load_model(self.model_dir, compile=False)
            backbone = model.get_layer('backbone')
            return lambda x: tf.squeeze(backbone(x[np.newaxis,...])).numpy()
        else:
            raise ValueError('invalid method')
    
    def __call__(self, audio):
        """take single track 4-second audio"""
        assert len(audio) == 64000
        return self.coder(self.preprocessor(audio))

