REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_rnn_agent import LatentRNNAgent
from .latent_oracle_rnn_agent import LatentOracleRNNAgent

from .latent_mse_rnn_agent import LatentMSERNNAgent
from .latent_cat_rnn_agent import LatentCatRNNAgent
from .latent_evolve_rnn_agent import LatentEvolveRNNAgent

from .latent_film_rnn_agent import LatentFiLMRNNAgent
from .latent_mixture_rnn_agent import LatentMixtureRNNAgent
from .latent_mixture_all_rnn_agent import LatentMixtureAllRNNAgent
from .latent_mixture_attention_rnn_agent import LatentMixtureAttentionRNNAgent
from .latent_input_rnn_agent import LatentInputRNNAgent
from .latent_mixture_input_rnn_agent import LatentMixtureInputRNNAgent
from .latent_mixture_input_3s5z_rnn_agent import LatentMixtureInput3s5zRNNAgent

from .mixture_role_rnn_agent import MixtureRoleRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_rnn"] = LatentRNNAgent
REGISTRY["latent_oracle_rnn"] = LatentOracleRNNAgent
REGISTRY["latent_cat_rnn"] = LatentCatRNNAgent
REGISTRY["latent_mse_rnn"] = LatentMSERNNAgent
REGISTRY["latent_evolve_rnn"] = LatentEvolveRNNAgent
REGISTRY["latent_film_rnn"] = LatentFiLMRNNAgent
REGISTRY["latent_mixture_rnn"] = LatentMixtureRNNAgent
REGISTRY["latent_mixture_all_rnn"] = LatentMixtureAllRNNAgent
REGISTRY["latent_mixture_attention_rnn"] = LatentMixtureAttentionRNNAgent
REGISTRY["latent_input_rnn"] = LatentInputRNNAgent
REGISTRY["latent_mixture_input_rnn"] = LatentMixtureInputRNNAgent
REGISTRY["latent_mixture_input_3s5z_rnn"] = LatentMixtureInput3s5zRNNAgent

REGISTRY["mixture_role_rnn"] = MixtureRoleRNNAgent
