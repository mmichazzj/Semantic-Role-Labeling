import theano
import theano.tensor as tensor
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np

from ..shared.numpy_utils import *
from ..shared.constants import RANDOM_SEED
from .util import *


def _slice(_x, n, dim):
  if _x.ndim == 3:
    return _x[:, :, n*dim : (n+1)*dim]
  return _x[:, n*dim : (n+1)*dim]

def _p(pp, name):
  return '%s_%s' % (pp, name)


class EmbeddingLayer(object):
  """ Embedding layer with concatenated features.
  """
  def __init__(self, embedding_shapes, embedding_inits=None, prefix='embedding'):
    self.embedding_shapes = embedding_shapes
    self.num_feature_types = len(embedding_shapes)
    self.output_size = sum([shape[1] for shape in self.embedding_shapes])
    print("Using {} feature types, projected output dim={}.".format(self.num_feature_types,
                                                                    self.output_size))
    self.embeddings = [get_variable(_p(prefix, i), shape, random_normal_initializer(0.0, 0.01))
                         for i, shape in enumerate(embedding_shapes)]
    # Initialize embeddings with pretrained values
    if embedding_inits != None:
      for emb, emb_init in zip(self.embeddings, embedding_inits):
        if emb_init != None:
          emb.set_value(numpy.array(emb_init, dtype=floatX))
    self.params = self.embeddings
  
  def connect(self, inputs):
    features = [None] * self.num_feature_types
    for i in range(self.num_feature_types):
      indices = inputs[:,:,i].flatten()
      proj_shape = [inputs.shape[0], inputs.shape[1], self.embedding_shapes[i][1]]
      features[i] = self.embeddings[i][indices].reshape(proj_shape)

    if self.num_feature_types == 1:
      return features[0]
    return tensor.concatenate(features, axis=2)
  

class SoftmaxLayer(object):
  def __init__(self, input_dim, label_space_size, prefix='softmax'):
    self.input_dim = input_dim
    self.label_space_size = label_space_size
    self.W = get_variable(_p(prefix, 'W'), [input_dim, self.label_space_size],
                          random_normal_initializer(0.0, 0.01))
    self.b = get_variable(_p(prefix, 'b'), [self.label_space_size],
                          all_zero_initializer())
    self.params = [self.W, self.b]
    
  def connect(self, inputs):
    energy = tensor.dot(inputs, self.W) + self.b
    energy = energy.reshape([energy.shape[0] * energy.shape[1], energy.shape[2]])
    log_scores = tensor.log(tensor.nnet.softmax(energy))
    predictions = tensor.argmax(log_scores, axis=-1)
    return (log_scores, predictions)


class CRFLayer(object):
  def __init__(self, input_dim, label_space_size, prefix='crf'):
    self.input_dim = input_dim
    self.label_space_size = label_space_size
    self.W = get_variable(_p(prefix, 'W'), [input_dim, self.label_space_size],
                          random_normal_initializer(0.0, 0.01))
    self.b = get_variable(_p(prefix, 'b'), [self.label_space_size],
                          all_zero_initializer())
    self.params = [self.W, self.b]
    

  def forward(self, observations, transitions, viterbi=False,
              return_alpha=False, return_best_sequence=False):
      """
      Takes as input:
          - observations, sequence of shape (n_steps, n_classes)
          - transitions, sequence of shape (n_classes, n_classes)
      Probabilities must be given in the log space.
      Compute alpha, matrix of size (n_steps, n_classes), such that
      alpha[i, j] represents one of these 2 values:
          - the probability that the real path at node i ends in j
          - the maximum probability of a path finishing in j at node i (Viterbi)
      Returns one of these 2 values:
          - alpha
          - the final probability, which can be:
              - the sum of the probabilities of all paths
              - the probability of the best path (Viterbi)
      """
      assert not return_best_sequence or (viterbi and not return_alpha)

      def recurrence(obs, previous, transitions):
          previous = previous.dimshuffle(0, 'x')
          obs = obs.dimshuffle('x', 0)
          if viterbi:
              scores = previous + obs + transitions
              out = scores.max(axis=0)
              if return_best_sequence:
                  out2 = scores.argmax(axis=0)
                  return out, out2
              else:
                  return out
          else:
              return log_sum_exp(previous + obs + transitions, axis=0)

      initial = observations[0]
      alpha, _ = theano.scan(
          fn=recurrence,
          outputs_info=(initial, None) if return_best_sequence else initial,
          sequences=[observations[1:]],
          non_sequences=transitions
      )

      if return_alpha:
          return alpha
      elif return_best_sequence:
          sequence, _ = theano.scan(
              fn=lambda beta_i, previous: beta_i[previous],
              outputs_info = tensor.cast(tensor.argmax(alpha[0][-1]), 'int32'),
              sequences = tensor.cast(alpha[1][::-1], 'int32')
          )
          sequence = tensor.concatenate([sequence[::-1], [tensor.argmax(alpha[0][-1])]])
          return sequence
      else:
          if viterbi:
              return alpha[-1].max(axis=0)
          else:
              return log_sum_exp(alpha[-1], axis=0)

  def connect(self, inputs):
    s_len = inputs.shape[0]

    energy = tensor.dot(inputs, self.W) + self.b
    energy = energy.reshape([energy.shape[0] * energy.shape[1], energy.shape[2]])
    transitions = shared((self.label_space_size, self.label_space_size), 'transitions')

    small = -1000
    b_s = np.array([[small] * self.label_space_size + [0, small]]).astype(np.float32)
    e_s = np.array([[small] * self.label_space_size + [small, 0]]).astype(np.float32)
    observations = tensor.concatenate(
        [energy, small * tensor.ones((s_len, 2))],
        axis=1
    )
    observations = tensor.concatenate(
        [b_s, observations, e_s],
        axis=0
    )

    # Score from tags
    tag_ids = tensor.ivector(name='tag_ids')
    real_path_score = energy[tensor.arange(s_len), tag_ids].sum()

    # Score from transitions
    b_id = theano.shared(value=np.array([self.label_space_size], dtype=np.int32))
    e_id = theano.shared(value=np.array([self.label_space_size + 1], dtype=np.int32))
    padded_tags_ids = tensor.concatenate([b_id, tag_ids, e_id], axis=0)
    real_path_score += transitions[
        padded_tags_ids[tensor.arange(s_len + 1)],
        padded_tags_ids[tensor.arange(s_len + 1) + 1]
    ].sum()

    log_scores, predictions = self.forward(observations, transitions)
    #cost = - (real_path_score - all_paths_scores)

    return (log_scores, predictions)

    
class CrossEntropyLoss(object):
  def connect(self, inputs, weights, labels):
    """ - inputs: flattened log scores from the softmax layer.
    """    
    y_flat = labels.flatten()
    x_flat_idx = tensor.arange(y_flat.shape[0])
    cross_ent = - inputs[x_flat_idx, y_flat].reshape([labels.shape[0], labels.shape[1]])
    if weights != None:
      cross_ent = cross_ent * weights
    # Summed over timesteps. Averaged across samples in the batch.
    return cross_ent.sum(axis=0).mean() 
    

class LSTMLayer(object):
  """ Basic LSTM. From the LSTM Tutorial.
  """
  def __init__(self, input_dim, hidden_dim, forget_bias = 1.0,
               input_dropout_prob = 0.0, recurrent_dropout_prob = 0.0,
               use_orthnormal_init = False,
               fast_predict=False,
               prefix='lstm'):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.forget_bias = forget_bias
    self.prefix = prefix
    self.fast_predict = fast_predict
    self._init_parameters(4, 4, use_orthnormal_init) 
    #self._init_dropout_layers(input_dropout_prob, recurrent_dropout_prob)

  def _init_parameters(self, num_i2h, num_h2h, use_orthnormal_init):
    """ num_i2h: Number of input-to-hidden projection blocks.
        num_h2h: Number of hidden-to-hidden projection blocks.
    """
    input_dim = self.input_dim
    hidden_dim = self.hidden_dim
    if use_orthnormal_init:
      self.W = get_variable(_p(self.prefix, 'W'), [input_dim, num_i2h * hidden_dim],
                            block_orth_normal_initializer([input_dim,], [hidden_dim] * num_i2h))
      self.U = get_variable(_p(self.prefix, 'U'), [hidden_dim, num_h2h * hidden_dim],
                            block_orth_normal_initializer([hidden_dim,], [hidden_dim] * num_h2h))
    else:
      self.W = get_variable(_p(self.prefix, 'W'), [input_dim, num_i2h * hidden_dim],
                            random_normal_initializer(0.0, 0.01))
      self.U = get_variable(_p(self.prefix, 'U'), [hidden_dim, num_h2h * hidden_dim],
                            random_normal_initializer(0.0, 0.01))
    self.b = get_variable(_p(self.prefix, 'b'), [num_i2h * hidden_dim], all_zero_initializer())
    self.params = [self.W, self.U, self.b]
    
  def _step(self, x_, m_, h_, c_):
    preact = tensor.dot(h_, self.U) + x_

    i = tensor.nnet.sigmoid(_slice(preact, 0, self.hidden_dim))
    f = tensor.nnet.sigmoid(_slice(preact, 1, self.hidden_dim) + self.forget_bias)
    o = tensor.nnet.sigmoid(_slice(preact, 2, self.hidden_dim))
    j = tensor.tanh(_slice(preact, 3, self.hidden_dim))

    c = f * c_ + i * j
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = o * tensor.tanh(c)
    #if self.recurrent_dropout_layer != None:
      #h = self.recurrent_dropout_layer.connect(h, self.is_train)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c
        
  def connect(self, inputs, mask, is_train):
    """ is_train: A boolean tensor.
    """
    max_length = inputs.shape[0]
    batch_size = inputs.shape[1]
    outputs_info = [tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim),
            tensor.alloc(numpy_floatX(0.), batch_size, self.hidden_dim)]
    # Dropout mask sharing for variational dropout.
    self.is_train = is_train
    #if self.recurrent_dropout_layer != None:
      #self.recurrent_dropout_layer.generate_mask([batch_size, self.hidden_dim], is_train)
    
    inputs = tensor.dot(inputs, self.W) + self.b
    rval, _ = theano.scan(self._step, # Scan function
                sequences=[inputs, mask], # Input sequence
                outputs_info=outputs_info,
                name=_p(self.prefix, '_layers'),
                n_steps=max_length) # scan steps
    return rval[0]