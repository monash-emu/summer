from computegraph import jaxify as _jaxify

_jaxify.set_using_jax(True)

from .adjust import Multiply, Overwrite
from .compartment import Compartment
from .model import CompartmentalModel
from .stratification import AgeStratification, StrainStratification, Stratification
