import os as _os

from . import util
from .util import build_model_with_jax

from jax.config import config as _jax_config

# Jax configuration
# FIXME: We need to find a more appropriate place to ensure this happens globally
_jax_config.update("jax_enable_x64", True)
_os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
