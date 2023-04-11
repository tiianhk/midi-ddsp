import os
import tensorflow as tf
import librosa
import numpy as np
import openl3
import joblib
import warnings
warnings.filterwarnings("ignore")


class TimbreCoder():
    """generate timbre embeddings."""

    def __init__(self, method):
        self.method = method
        self.model_dir = os.path.join('./timbre_encoding/models/', method)
        # to-do: store centroids of midi-ddsp
        self.centroids = np.load(os.path.join('./timbre_encoding/centroids/', method+'.npy'))
        self.preprocessor = self._load_preprocessor()
        self.coder = self._load_coder()
        self.sample_rate = 16000
        self.ndim = self._ndim_dict()[method]

    def _ndim_dict(self):
        return {'lda': 12, 'openl3': 512, 'flat_triplet': 64, 'hierarchical_triplet': 64, 'midi-ddsp': 64}

    def _load_preprocessor(self):
        if self.method == 'lda':
            """time averaged mfcc"""
            return lambda y: np.mean(librosa.feature.mfcc(y=y, sr=self.sample_rate), axis=-1)
        elif self.method == 'openl3':
            return None
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """melspectrogram"""
            return lambda y: librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, power=1)[...,np.newaxis]
        elif self.method == 'midi-ddsp':
            return None
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
            return lambda x: np.mean(openl3.get_audio_embedding(x, sr=self.sample_rate,
                model=model, center=False, hop_size=0.5, batch_size=128, verbose=0)[0], axis=-2)
        elif self.method == 'flat_triplet' or self.method == 'hierarchical_triplet':
            """pretrained triplet models"""
            model = tf.keras.models.load_model(self.model_dir, compile=False)
            backbone = model.get_layer('backbone')
            return lambda x: backbone(x).numpy()
        elif self.method == 'midi-ddsp':
            return None
        else:
            raise ValueError('invalid method')

    def get_embedding_from_audio(self, audio):
        assert self.method != 'midi-ddsp'
        assert len(audio.shape)<=2 and audio.shape[-1]==64000
        self.batch = len(audio.shape)==2
        if isinstance(audio, tf.Tensor):
            audio = audio.numpy()
        """preprocessing"""
        if self.batch:
            if self.method == 'openl3':
                feature = [audio[i] for i in range(audio.shape[0])]
            else:
                feature = np.stack([self.preprocessor(audio[i]) for i in range(audio.shape[0])])
        else:
            if self.method == 'openl3':
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
        print('timbre coder is called.')
        if audio is not None:
            print(f'processing audio with shape {audio.shape}')
            return self.get_embedding_from_audio(audio)
        elif inst is not None:
            assert inst < len(self.centroids)
            return self.centroids[inst][np.newaxis,...]
        else:
            assert inst1 < len(self.centroids) and inst2 < len(self.centroids)
            assert interp_ratio > 0 and interp_ratio < 1
            emb = interp_ratio * self.centroids[inst1] + (1 - interp_ratio) * self.centroids[inst2]
            return emb[np.newaxis,...]

