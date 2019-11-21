REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_rnn_agent import LatentRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_rnn"] = LatentRNNAgent
