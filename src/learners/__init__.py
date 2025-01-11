from .q_learner import QLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner

from .dmcg_learner import DMCGLearner
REGISTRY["dmcg_learner"] = DMCGLearner
