REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_rnn_agent import LatentRNNAgent
from .latent_mixture_rnn_agent import LatentMixtureRNNAgent
from .latent_input_rnn_agent import LatentInputRNNAgent
from .latent_mixture_input_rnn_agent import LatentMixtureInputRNNAgent
from .latent_mixture_input_3s5z_rnn_agent import LatentMixtureInput3s5zRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_rnn"] = LatentRNNAgent
REGISTRY["latent_mixture_rnn"] = LatentMixtureRNNAgent
REGISTRY["latent_input_rnn"] = LatentInputRNNAgent
REGISTRY["latent_mixture_input_rnn"] = LatentMixtureInputRNNAgent
REGISTRY["latent_mixture_input_3s5z_rnn"] = LatentMixtureInput3s5zRNNAgent
