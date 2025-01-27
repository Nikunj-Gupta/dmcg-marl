REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

from .dmcg_agent import DMCGAgent
REGISTRY["dmcg"] = DMCGAgent
from .dmcg_agent_feat import DMCGFeatureAgent
REGISTRY["dmcg_feat"] = DMCGFeatureAgent 