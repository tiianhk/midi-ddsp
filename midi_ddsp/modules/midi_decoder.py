#  Copyright 2022 The MIDI-DDSP Authors.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""The decoder wrappers for predict synthesis parameters from MIDI or from
MIDI and note expression controls."""

import tensorflow as tf
import ddsp
import ddsp.training
import numpy as np
from .interpretable_conditioning import get_interpretable_conditioning, \
  get_conditioning_dict
from midi_ddsp.data_handling.instrument_name_utils import NUM_INST

tfk = tf.keras
tfkl = tfk.layers


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(index, d_model):
  angle_rads = get_angles(index,
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads

  return tf.cast(pos_encoding, tf.float32)


class ExpressionMidiDecoder(tfkl.Layer):
  """Decoder wrapper for predicting synthesis parameters from MIDI and
  note expression controls."""

  def __init__(self, decoder, z_preconditioning_stack, multi_instrument=False,
               position_code='none', decoder_type='dilated_conv',
               without_note_expression=False, timbre_coder=None):
    self.decoder_type = decoder_type
    self.z_preconditioning_stack = z_preconditioning_stack
    self.decoder = decoder
    self.position_code = position_code
    self.multi_instrument = multi_instrument
    self.timbre_coder = timbre_coder
    if multi_instrument:
      if timbre_coder is None or timbre_coder.method == 'midi_ddsp':
        self.instrument_emb = tfkl.Embedding(NUM_INST, 64)
      else:
        if timbre_coder.ndim != 64:
          self.instrument_emb_proj = tfkl.Dense(64)
    self.without_note_expression = without_note_expression
    super().__init__()

  def gen_params_from_cond(self, conditioning_dict, midi_features,
                           instrument_id=None, audio=None, synth_params=None,
                           training=False, display_progressbar=False):
    q_pitch, q_vel, f0_loss_weights, onsets, offsets = midi_features
    # note-wise conditioning

    if self.without_note_expression:
      z_conditioning = tf.concat([q_pitch / 127,
                                  tf.cast(onsets, tf.float32)[..., tf.newaxis],
                                  tf.cast(offsets, tf.float32)[..., tf.newaxis]
                                  ], -1)
    else:
      z_conditioning = tf.stop_gradient(
        tf.concat(list(conditioning_dict.values()), -1))
      z_conditioning = tf.concat([z_conditioning,
                                  q_pitch / 127,
                                  tf.cast(onsets, tf.float32)[..., tf.newaxis],
                                  tf.cast(offsets, tf.float32)[..., tf.newaxis]
                                  ], -1)

    if self.position_code == 'index_length':
      note_mask = ddsp.training.nn.get_note_mask_from_onset(q_pitch, onsets)
      each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                             tf.float32)
      each_note_len = tf.reduce_max(each_note_idx, axis=1,
                                    keepdims=True) * tf.cast(each_note_idx > 0,
                                                             tf.float32)
      each_note_idx = tf.reduce_sum(each_note_idx, axis=-1)[..., tf.newaxis]
      each_note_len = tf.reduce_sum(each_note_len, axis=-1)[..., tf.newaxis]
      relative_position = tf.math.divide_no_nan(each_note_idx, each_note_len)
      z_conditioning = tf.concat([z_conditioning, relative_position], -1)
    elif self.position_code == 'sinusoidal':
      note_mask = ddsp.training.nn.get_note_mask_from_onset(q_pitch, onsets)
      each_note_idx = tf.cumsum(note_mask, axis=1) * tf.cast(~(note_mask == 0),
                                                             tf.float32)
      each_note_idx = tf.reduce_sum(each_note_idx, axis=-1)[..., tf.newaxis]
      pos_code = positional_encoding(each_note_idx.numpy().astype(np.int64), 64)
      z_conditioning = tf.concat([z_conditioning, pos_code], -1)

    # --- Precondition
    z_midi_decoder = self.z_preconditioning_stack(z_conditioning)
    """timbre encoding"""
    if self.multi_instrument:
      unpooled_flag = False
      if self.timbre_coder is None:
        if isinstance(instrument_id, int):
          instrument_id = tf.constant([instrument_id])
        inst_emb = self.instrument_emb(instrument_id)
      else:
        if audio is not None:
          """autoencoder"""
          if self.timbre_coder.ndim == 64:
            if self.timbre_coder.method == 'midi_ddsp':
              print('Warning: this timbre coder only contains centroids. '
                    'But it is asked to process audio input. '
                    'This should only appear when activating interpolation '
                    'for model trained without timbre encoding.')
              inst_emb = self.timbre_coder(inst=int(instrument_id))
            else:
              inst_emb = self.timbre_coder(audio=audio)
          else:
            inst_emb = self.instrument_emb_proj(self.timbre_coder(audio=audio))
        else:
          """synthesis"""
          if isinstance(instrument_id, int):
            """centroid"""
            inst_emb = self.timbre_coder(inst=instrument_id)
          elif isinstance(instrument_id, tuple):
            """interpolate between centroids"""
            inst1, inst2, interp_ratio, is_gradual = instrument_id
            if is_gradual == False:
              inst_emb = self.timbre_coder(inst1=inst1, inst2=inst2, interp_ratio=interp_ratio)
            else:
              unpooled_flag = True
              inst_emb_1 = self.timbre_coder(inst=inst1)
              inst_emb_2 = self.timbre_coder(inst=inst2)
              inst_emb = tf.linspace(inst_emb_1, inst_emb_2, z_midi_decoder.shape[1], axis=1)
              inst_emb = tf.linalg.normalize(inst_emb[0], axis=1)[0][tf.newaxis,...]
          if self.timbre_coder.ndim != 64:
            inst_emb = self.instrument_emb_proj(inst_emb)
      if unpooled_flag:
        instrument_z = inst_emb
      else:
        instrument_z = tf.tile(
          inst_emb[:, tf.newaxis, :], [1, z_midi_decoder.shape[1], 1])
      z_midi_decoder = tf.concat([z_midi_decoder, instrument_z], -1)

    # --- MIDI Decoding
    if self.decoder_type == 'dilated_conv':
      params_pred = self.decoder(q_pitch, q_vel, z_midi_decoder)
    elif self.decoder_type == 'noise_dilated_conv':
      noise = tf.random.normal(
        [z_midi_decoder.shape[0], z_midi_decoder.shape[1], 100])
      params_pred = self.decoder(noise, q_pitch, z_midi_decoder)
    elif 'rnn' in self.decoder_type:
      params_pred = self.decoder(q_pitch, z_midi_decoder, conditioning_dict,
                                 out=synth_params, training=training,
                                 display_progressbar=display_progressbar)
    # midi_decoder: [q_pitch, z_midi_decoder] -> synth params
    return z_midi_decoder, params_pred

  def gen_cond_dict(self, synth_params_normalized, midi_features):
    f0, amps, hd, noise = synth_params_normalized
    q_pitch, q_vel, f0_loss_weights, onsets, offsets = midi_features

    f0_midi_gt = ddsp.core.midi_to_hz(q_pitch, midi_zero_silence=True)
    conditioning = get_interpretable_conditioning(f0_midi_gt, f0, amps, hd,
                                                  noise)

    # --- Z Note Encoding
    conditioning_dict = get_conditioning_dict(conditioning, q_pitch, onsets,
                                              pool_type='note_pooling')
    return conditioning_dict

  def call(self, features, synth_params_normalized, midi_features,
           training=False, synth_params=None):
    conditioning_dict = self.gen_cond_dict(synth_params_normalized,
                                           midi_features)

    instrument_id = features['instrument_id'] if self.multi_instrument else None
    audio = features['audio'] if self.timbre_coder is not None else None

    if self.decoder_type == 'rnn_f0_ld':
      synth_params = features
    z_midi_decoder, params_pred = self.gen_params_from_cond(conditioning_dict,
                                                            midi_features,
                                                            instrument_id=
                                                            instrument_id,
                                                            audio=audio,
                                                            synth_params=
                                                            synth_params,
                                                            training=training)

    params_pred['z_midi_decoder'] = z_midi_decoder

    return conditioning_dict, params_pred


class MidiDecoder(tfkl.Layer):
  """Decoder wrapper for predicting synthesis parameters from only MIDI."""

  def __init__(self, decoder, multi_instrument=False):
    super().__init__()
    self.decoder = decoder
    self.multi_instrument = multi_instrument
    if multi_instrument:
      self.instrument_emb = tfkl.Embedding(NUM_INST, 64)
    self.pitch_emb = tfkl.Embedding(128, 64)

  def call(self, features, synth_params_normalized, midi_features,
           training=False, synth_params=None):
    instrument_id = features['instrument_id'] if self.multi_instrument else None
    q_pitch, q_vel, f0_loss_weights, onsets, offsets = midi_features
    instrument_z = tf.tile(self.instrument_emb(instrument_id)[:, tf.newaxis, :],
                           [1, q_pitch.shape[1], 1])
    z_midi_decoder = tf.concat(
      [self.pitch_emb(tf.cast(q_pitch, tf.int64)[..., 0]),
       tf.cast(onsets, tf.float32)[..., tf.newaxis],
       tf.cast(offsets, tf.float32)[..., tf.newaxis],
       instrument_z],
      -1)  # HACK
    params_pred = self.decoder(q_pitch, z_midi_decoder, out=features,
                               training=training)
    return {}, params_pred
