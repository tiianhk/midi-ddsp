import numpy as np


def get_embedding(audio, coder):
    """convert single audio or batched audio to timbre embeddings."""
    assert len(audio.shape)<=2 and audio.shape[-1]==64000
    audio = np.array(audio)
    if len(audio.shape) == 1:
        return coder(audio)
    else:
        timbre_emb = []
        for i in range(audio.shape[0]):
            timbre_emb.append(coder(audio[i]))
        return np.array(timbre_emb)
