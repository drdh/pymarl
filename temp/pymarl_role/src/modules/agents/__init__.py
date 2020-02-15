REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_dis_rnn_agent import LatentDisRNNAgent
from .latent_ce_dis_rnn_agent import LatentCEDisRNNAgen

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_dis_rnn"] = LatentDisRNNAgent
REGISTRY["latent_ce_dis_rnn"] = LatentCEDisRNNAgen
