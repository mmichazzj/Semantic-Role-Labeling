from .constants import *
from .dictionary import Dictionary
from .reader import *

def get_srl_features(sentences, config, feature_dicts=None):
  ''' TODO: Support adding more features.
  '''
  feature_names = config.features
  feature_sizes = config.feature_sizes
  use_se_marker = config.use_se_marker
  
  features = []
  feature_shapes = []
  offset = int(use_se_marker)
  for fname, fsize in zip(feature_names, feature_sizes):
    if fname == "pred":
      features.append([[int(i == sent[1] + offset) for i in range(len(sent[0]))] for sent in sentences])
    elif fname == 'argu':
      features.append([[int(i + offset) for i in range(len(sent[0]))] for sent in sentences])
    elif fname == 'ctx-p':
      features.append([[int(i in range(sent[1] + offset - config.pred_content_len, sent[1] + offset + config.pred_content_len + 1))\
        for i in range(len(sent[0]))] for sent in sentences])
    feature_shapes.append([4, fsize])  #4 features
 
  return (list(zip(*features)), feature_shapes)
  #return (list(zip(features[0], features[1], features[2])), feature_shapes)

