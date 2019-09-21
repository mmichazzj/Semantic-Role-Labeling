import numpy
import theano
from theano import config as theano_config
import theano.tensor as tensor
from neural_srl.shared.numpy_utils import random_normal_initializer


floatX = theano_config.floatX
def numpy_floatX(data):
  return numpy.asarray(data, dtype=floatX)

def get_variable(name, shape, initializer=None, dtype=floatX):
  if initializer != None:
    param = initializer(shape, dtype)
  else:
    param = random_normal_initializer()(shape, dtype)

  return theano.shared(value=param, name=name, borrow=True)

def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = numpy.zeros(shape)  # bias are initialized with zeros
    else:
        drange = numpy.sqrt(6. / (numpy.sum(shape)))
        value = drange * numpy.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(floatX), name=name)

def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + tensor.log(tensor.exp(x - xmax).sum(axis=axis))