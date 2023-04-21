import os
import tensorflow as tf
import librosa
import numpy as np
import openl3
import joblib
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")


class TimbreCoder():
    """generate timbre embeddings."""

    def __init__(self, method):
        self.method = method
        self.model_dir = os.path.join('./timbre_encoding/models/', method)
        centroids_filename = 'openl3' if method == 'openl3_precompute' else method
        self.centroids = np.float32(np.load(os.path.join(
            './timbre_encoding/centroids/', centroids_filename+'.npy')))
        self.preprocessor = self._load_preprocessor()
        self.coder = self._load_coder()
        self.sample_rate = 16000
        self.ndim = self._ndim_dict()[method]

    def _ndim_dict(self):
        return {'lda': 12, 
                'openl3': 512, 
                'openl3_precompute': 512, 
                'flat_triplet': 64, 
                'hierarchical_triplet': 64, 
                'midi_ddsp': 64}

    def _load_preprocessor(self):
        if self.method == 'lda':
            """time averaged mfcc"""
            return lambda y: np.mean(librosa.feature.mfcc(y=y, sr=self.sample_rate), axis=-1)
        elif self.method in ['openl3', 'openl3_precompute', 'midi_ddsp']:
            return None
        elif self.method in ['flat_triplet', 'hierarchical_triplet']:
            """melspectrogram"""
            return lambda y: librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, power=1)[...,np.newaxis]
        else:
            raise ValueError('invalid method')

    def _load_coder(self):
        if self.method == 'lda':
            """pretrained standardization and LDA transformation"""
            model = joblib.load(self.model_dir+'.joblib')
            return lambda x: model.transform(x)
        elif self.method == 'openl3':
            """time averaged openl3 embedding"""
            model = openl3.models.load_audio_embedding_model(
                input_repr="mel128", content_type="music", embedding_size=512)
            model.trainable = False
            return lambda x: np.mean(openl3.get_audio_embedding(x, sr=self.sample_rate,
                model=model, center=False, hop_size=0.5, batch_size=128, verbose=0)[0], axis=-2)
        elif self.method == 'openl3_precompute':
            raise NotImplementedError('this method have not passed the test yet, do not use it.')
            """pre-computed openl3 embedding"""
            self.audio_mtx = np.load('timbre_encoding/precompute/audio.npy')
            self.emb_mtx = np.load('timbre_encoding/precompute/openl3_emb.npy')
            def query_for_emb(x):
                emb = []
                for audio in x:
                    idx = np.where(np.apply_along_axis(np.array_equal, 1, self.audio_mtx, audio))[0]
                    if idx.shape[0] > 0:
                        emb.append(self.emb_mtx[idx[0]])
                    else:
                        new_emb = np.mean(openl3.get_audio_embedding(audio, sr=self.sample_rate, 
                            input_repr="mel128", content_type="music", embedding_size=512, 
                            center=False, hop_size=0.5, batch_size=128, verbose=0)[0], axis=-2)
                        self.audio_mtx = np.concatenate((self.audio_mtx, audio[np.newaxis,...]))
                        self.emb_mtx = np.concatenate((self.emb_mtx, new_emb[np.newaxis,...]))
                        emb.append(new_emb)
                return np.stack(emb)
            return lambda x: query_for_emb(x)
        elif self.method in ['flat_triplet', 'hierarchical_triplet']:
            """pretrained triplet models"""
            model = tf.keras.models.load_model(self.model_dir, compile=False)
            backbone = model.get_layer('backbone')
            backbone.trainable = False
            return lambda x: normalize(backbone(x).numpy())
        elif self.method == 'midi_ddsp':
            return None
        else:
            raise ValueError('invalid method')

    def get_embedding_from_audio(self, audio):
        assert len(audio.shape)<=2 and audio.shape[-1]==64000
        self.batch = len(audio.shape)==2
        if isinstance(audio, tf.Tensor):
            audio = audio.numpy()
        """preprocessing"""
        if self.batch:
            if self.method in ['openl3', 'openl3_precompute']:
                feature = [audio[i] for i in range(audio.shape[0])]
            else:
                feature = np.stack([self.preprocessor(audio[i]) for i in range(audio.shape[0])])
        else:
            if self.method in ['openl3', 'openl3_precompute']:
                feature = [audio]
            else:
                feature = self.preprocessor(audio)[np.newaxis,...]
        """encoding"""
        return self.coder(feature)

    def __call__(self, audio=None, inst=None,
                 inst1=None, inst2=None, interp_ratio=None):
        """
            Args:
                audio: np.array or tf.Tensor,
                    single or batched 4-second audio,
                    shape (64000,) or (n, 64000),
                    audio is converted to embeddings by a model
                inst_id: retrieve the centroid embedding for the instrument
                inst_a_id, inst_b_id, interp_ratio:
                    get the linear interpolation between two centroids
            Returns:
                embedding: np.array,
                    'lda' shape (n, 12),
                    'openl3' shape (n, 512),
                    'flat_triplet' or 'hierarchical_triplet' shape (n, 64)
        """
        if audio is not None:
            return self.get_embedding_from_audio(audio)
        elif inst is not None:
            assert inst < len(self.centroids)
            return self.centroids[inst][np.newaxis,...]
        else:
            assert inst1 < len(self.centroids) and inst2 < len(self.centroids)
            assert interp_ratio > 0 and interp_ratio < 1
            emb = interp_ratio * self.centroids[inst1] + (1 - interp_ratio) * self.centroids[inst2]
            emb = emb[np.newaxis,...]
            if self.method in ['flat_triplet', 'hierarchical_triplet']:
                emb = normalize(emb)
            return emb

