REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_rnn_agent import LatentRNNAgent
from .latent_input_rnn_agent import LatentInputRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_rnn"] = LatentRNNAgent
REGISTRY["latent_input_rnn"] = LatentInputRNNAgent
