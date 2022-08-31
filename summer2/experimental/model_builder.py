from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from summer2 import CompartmentalModel

from inspect import getfullargspec
from typing import Union, Any

from collections.abc import Iterable
from numbers import Real

# from pydantic import BaseModel
# from pydantic.main import ModelMetaclass

from summer2.parameters import Parameter, Function, Data
from computegraph import ComputeGraph
from computegraph.utils import expand_nested_dict, is_var

GraphObj = Union[Function, Data]


class ParamStruct:
    pass


class ModelBuilder:
    def __init__(self, params: dict, param_class: type):
        self._params = params
        self._params_expanded = expand_nested_dict(params, include_parents=True)
        self.params = self._pyd_params = param_class(**params)
        label_parameters(self.params, params)

    def set_model(self, model: CompartmentalModel):
        self.model = model
        model.builder = self

    def _get_func_args(self, key):
        return [Parameter(k) for k in [*self._params_expanded[key]]]

    def _get_func_kwargs(self, key):
        return {k: Parameter(k) for k in [*self._params_expanded[key]]}

    def find_key_from_obj(self, obj):
        return find_key_from_obj(obj, self.params, self._params, None)

    def find_obj_from_key(self, key):
        return find_obj_from_key(key, self.params)

    def get_param(self, key: str, create: bool = False):
        """Get a Parameter (computegraph Variable) for the given key
        If this key is not contained in the initial parameters, register it
        as a required additional parameter (or raise an Exception if create is False)

        Args:
            key: Key of the parameter
            create: Add this parameter as an output if required - will raise an exception
                    if False and key is not already present in the input parameters

        Returns:
            computegraph Parameter
        """
        if isinstance(key, str):
            key = key
        elif is_var(key, "parameters"):
            key = key.name
        else:
            key = find_key_from_obj(key, self.params, self._params_expanded)
        if key not in self._params_expanded:
            if create:
                if key not in self.required_outputs:
                    self.required_outputs.add(key)
            else:
                raise KeyError(f"Parameter {key} not found in input parameters")
        return Parameter(key)

    def _get_value(self, key: str):
        """Return the initial parameter value for the given key

        Args:
            key (str): Parameter key
        """
        if key in self._params_expanded:
            return self._params_expanded[key]
        else:
            raise KeyError("Key not found in initial parameters", key)

    def get_mapped_func(self, func: callable, param_obj: ParamStruct, kwargs=None):
        argspec = getfullargspec(func)

        kwargs = {} if kwargs is None else kwargs

        supplied_argkeys = list(kwargs)

        msg = f"Function arguments {argspec} do not match pydantic object {param_obj}"
        assert all(
            [hasattr(param_obj, arg) for arg in argspec.args if arg not in supplied_argkeys]
        ), msg
        base_key = self.find_key_from_obj(param_obj)
        mapped_args = {arg: Parameter(f"{base_key}.{arg}") for arg in argspec.args}
        mapped_args.update(kwargs)
        return Function(func, [], mapped_args)

    def get_jax_runner(self, jit=True, solver=None):
        self.model.finalize()
        run_everything, _, _ = get_full_runner(self, True, solver=solver)
        if jit:
            from jax import jit as _jit

            run_everything = _jit(run_everything)
        return run_everything

    def get_input_parameters(self):
        return self.model.get_input_parameters()

    def get_default_parameters(self) -> dict:
        default_params = {
            k: v for k, v in self._params_expanded.items() if k in self.get_input_parameters()
        }
        return default_params


def label_parameters(pstruct: ParamStruct, pdict: dict, layer: list = None):
    """Walk the parameter dictionary pdict, alongside the ParamStruct,
       setting concrete parameter names for each AbstractParameter encountered

    Args:
        pstruct (ParamStruct): The ParamStruct describing the model
        pdict (dict): The parameter dictionary used as constructor for pstruct
        layer: The current naming layer (used internally, do not set)
    """
    if layer is None:
        layer = []
    for k, v in pdict.items():
        if isinstance(pstruct, dict):
            cur_pstruct = pstruct[k]
        else:
            cur_pstruct = getattr(pstruct, k)
        if isinstance(cur_pstruct, Parameter):
            param_key = ".".join(layer + [k])
            cur_pstruct.set_key(param_key)
        elif isinstance(cur_pstruct, dict):
            label_parameters(cur_pstruct, v, layer + [k])
        elif isinstance(cur_pstruct, ParamStruct):
            assert isinstance(v, dict)
            label_parameters(cur_pstruct, v, layer + [k])


def find_key_from_obj(obj: Any, pydparams: ParamStruct, params: dict, layer=None, is_dict=False):
    if layer is None:
        layer = []
    for k, v in params.items():

        if is_dict:
            cur_pydobj = pydparams[k]
        else:
            cur_pydobj = getattr(pydparams, k)
        if cur_pydobj is obj:
            if isinstance(cur_pydobj, ParamStruct) or isinstance(cur_pydobj, Parameter):
                return ".".join(layer + [k])
            else:
                raise TypeError("Cannot match against type", type(obj))
        if isinstance(cur_pydobj, dict):
            res = find_key_from_obj(obj, cur_pydobj, v, layer + [k], True)
            if res is not None:
                return res
        elif isinstance(cur_pydobj, ParamStruct):
            assert isinstance(v, dict)
            res = find_key_from_obj(obj, cur_pydobj, v, layer + [k])
            if res is not None:
                return res

    if len(layer) == 0:
        raise Exception(f"Unable to match {obj} in parameters dictionary", obj, pydparams, layer)


def find_obj_from_key(key: str, pydparams: ParamStruct) -> Any:
    """Find the matching object within a BaseModel for a given key

    Args:
        key: The expanded parameter key
        pydparams: The full instantiated BaseModel

    Raises:
        TypeError: Raised if uniqueness of object cannot be guaranteed

    Returns:
        The corresponding object within the BaseModel
    """
    cur_obj = pydparams
    for layer in key.split("."):
        if isinstance(cur_obj, ParamStruct):
            cur_obj = getattr(cur_obj, layer)
        elif isinstance(cur_obj, Iterable):
            cur_obj = cur_obj[layer]
        else:
            raise TypeError("Cannot resolve for type", key, cur_obj, type(cur_obj))
    return cur_obj


def get_full_runner(builder, use_jax=False, solver=None):
    if use_jax:
        from summer2.runner.jax.util import get_runner
        from summer2.runner.jax.model_impl import build_run_model

        jrunner = get_runner(builder.model)
        jax_run_func, jax_runner_dict = build_run_model(jrunner, solver=solver)

    model_input_p = builder.model.get_input_parameters()

    def run_everything(parameters=None, **kwargs):

        params_base = builder._params_expanded.copy()
        if parameters is not None:
            params_base.update(parameters)

        model_params = {k: v for k, v in params_base.items() if k in model_input_p}

        if use_jax:
            return jax_run_func(parameters=model_params)
        else:
            builder.model.run(parameters=model_params, **kwargs)
            return builder.model

    if use_jax:
        return run_everything, jax_run_func, jax_runner_dict
    else:
        return run_everything


def is_real(v):
    return isinstance(v, Real)


def parameter_class(constraint_func=is_real, desc: str = None, full_desc: str = None):

    _desc = desc
    _full_desc = full_desc

    class ConcreteParameter(Parameter):

        constraint = constraint_func
        description = _desc
        full_description = _full_desc

        def __init__(self, value):
            super().__init__(key=None)
            self.value = value

        def __repr__(self):
            pkey = self.key or self.description or "param"
            return f"{pkey}({self.value})"

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v):
            if not isinstance(v, Real):
                raise TypeError("Real required")
            if not cls.constraint(v):
                raise ValueError(f"Constraint failed", cls.constraint, v)
            return cls(v)

    return ConcreteParameter
