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
            return None
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """melspectrogram with channel dimension"""
            return lambda y: librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, power=1)[...,np.newaxis]
        else:
            raise ValueError('invalid method')
    
    def _load_coder(self):
        if self.method == 'lda':
            """pretrained standardization and LDA transformation"""
            model = joblib.load(self.model_dir+'.joblib')
            return lambda x: np.squeeze(model.transform(x))
        elif self.method == 'openl3':
            """time averaged openl3 embedding"""
            model = openl3.models.load_audio_embedding_model(
                input_repr="mel128", content_type="music", embedding_size=512)
            return lambda x: np.mean(openl3.get_audio_embedding(x, sr=self.sample_rate,
                model=model, center=False, hop_size=0.5, batch_size=128, verbose=0)[0], axis=-2)
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """pretrained triplet models"""
            model = tf.keras.models.load_model(self.model_dir, compile=False)
            backbone = model.get_layer('backbone')
            return lambda x: tf.squeeze(backbone(x)).numpy()
        else:
            raise ValueError('invalid method')
    
    def get_embedding(self, audio):
        assert len(audio.shape)<=2 and audio.shape[-1]==64000
        self.batch = len(audio.shape)==2
        if isinstance(audio, tf.Tensor):
            audio = audio.numpy()
        if self.batch:
            if self.method == 'openl3':
                # openl3 takes list of audio as input
                feature = [audio[i] for i in range(audio.shape[0])]
            else:
                # use librosa to preprocess one track at a time
                feature = np.stack([self.preprocessor(audio[i]) for i in range(audio.shape[0])])
        else:
            if self.method == 'openl3':
                feature = audio
            else:
                # add batch dimension for single track
                feature = self.preprocessor(audio)[np.newaxis,...]
        return self.coder(feature)

    def __call__(self, audio):
        """
            Args:
                audio: np.array or tf.Tensor,
                    single or batched 4-second audio,
                    shape (64000,) or (n, 64000)
            Returns:
                embedding: np.array,
                    'lda' shape (12,) or (n, 12)
                    'openl3' shape (512,) or (n, 512)
                    'flat_triplet' or 'hierarchical_triplet' shape (64,) or (n, 64)
        """
        return self.get_embedding(audio)

