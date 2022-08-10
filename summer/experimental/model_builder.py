from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from summer import CompartmentalModel

from inspect import getfullargspec
from typing import Union, Any

from collections.abc import Iterable
from numbers import Real

# from pydantic import BaseModel
# from pydantic.main import ModelMetaclass

from summer.parameters import Parameter, Function, Data
from computegraph import ComputeGraph
from computegraph.utils import expand_nested_dict, is_var

from .abstract_parameter import AbstractParameter

GraphObj = Union[Function, Data]


class ParamStruct:
    pass


class ModelBuilder:
    def __init__(self, params: dict, param_class: ParamStruct):
        self._params = params
        self._params_expanded = expand_nested_dict(params, include_parents=True)
        self.params = self._pyd_params = param_class(**params)
        self.input_graph = {}

        self.required_outputs = set()

    def add_output(self, key: str, graph_obj: GraphObj):
        if key in self.input_graph:
            raise Exception(f"Key {key} already exists in graph as {self.input_graph[key]}")
        self.input_graph[key] = graph_obj

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
                    if False if key is not already present in the input parameters

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

    def get_output(self, key):
        if key not in self.input_graph:
            raise KeyError(f"{key} does not exist in builder outputs")
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

    def get_jax_runner(self, jit=True):
        self.model.finalize()
        run_everything, run_inputs, _, _ = get_full_runner(self, True)
        if jit:
            from jax import jit as _jit

            run_everything = _jit(run_everything)
        return run_everything


def find_key_from_obj(obj: Any, pydparams: ParamStruct, params: dict, layer=None, is_dict=False):
    if layer is None:
        layer = []
    for k, v in params.items():

        if is_dict:
            cur_pydobj = pydparams[k]
        else:
            cur_pydobj = getattr(pydparams, k)
        if cur_pydobj is obj:
            if isinstance(cur_pydobj, ParamStruct) or isinstance(cur_pydobj, AbstractParameter):
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


def get_full_runner(builder, use_jax=False):
    graph_run = ComputeGraph(builder.input_graph).get_callable()

    if use_jax:
        from summer.runner.jax.util import get_runner
        from summer.runner.jax.model_impl import build_run_model

        jrunner = get_runner(builder.model)
        jax_run_func, jax_runner_dict = build_run_model(jrunner)

    model_input_p = builder.model.get_input_parameters()

    def run_everything(parameters=None, **kwargs):

        params_base = builder._params_expanded.copy()
        if parameters is not None:
            params_base.update(parameters)

        graph_outputs = graph_run(parameters=params_base)
        params_base.update(graph_outputs)

        model_params = {k: v for k, v in params_base.items() if k in model_input_p}

        if use_jax:
            return jax_run_func(parameters=model_params)
        else:
            builder.model.run(parameters=model_params, **kwargs)
            return builder.model

    def run_inputs(param_updates=None):
        parameters = builder._params_expanded.copy()
        if param_updates is not None:
            parameters.update(param_updates)

        graph_outputs = graph_run(parameters=parameters)
        parameters.update(graph_outputs)

        model_params = {k: v for k, v in parameters.items() if k in model_input_p}
        return model_params

    if use_jax:
        return run_everything, run_inputs, jax_run_func, jax_runner_dict
    else:
        return run_everything


def is_real(v):
    return isinstance(v, Real)


def parameter_class(constraint_func=is_real, description: str = None):

    _description = description

    class ConcreteParameter(AbstractParameter):

        constraint = constraint_func
        description = _description

        def __init__(self, value):
            self.value = value

        def __repr__(self):
            if self.description:
                desc_str = f" {self.description} "
            else:
                desc_str = ""
            return f"Parameter:{desc_str}({self.value})"

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
