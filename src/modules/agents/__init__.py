REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_rnn_agent import LatentRNNAgent
from .latent_input_rnn_agent import LatentInputRNNAgent
from .latent_mixture_input_rnn_agent import LatentMixtureInputRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_rnn"] = LatentRNNAgent
REGISTRY["latent_input_rnn"] = LatentInputRNNAgent
REGISTRY["latent_mixture_input_rnn"] = LatentMixtureInputRNNAgent
