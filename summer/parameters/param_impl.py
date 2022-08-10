from typing import Tuple, List, Iterable, Any
from numbers import Real

from summer.parameters.params import build_args, is_var, Function, Variable, ComputedValue, Time
from summer.experimental.model_builder import AbstractParameter, ModelBuilder
from summer.experimental.abstract_parameter import LazyParameter, set_keys


class ModelParameter:
    def get_value(self, time: float, computed_values: dict, parameters: dict):
        raise NotImplementedError

    def __eq__(self, other):
        return hash(other) == hash(self)

    def is_time_varying(self):
        return False


class DataParameter(ModelParameter):
    """Generic container for Python objects"""

    def __init__(self, value):
        self.value = value

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"DataParameter: {self.value}"


class FloatParameter(ModelParameter):
    def __init__(self, value):
        self.value = value

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"FloatParameter: {self.value}"


class GraphParameter(ModelParameter):
    def __init__(self, name):
        self.name = name

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return parameters[self.name]

    def __hash__(self):
        return hash((self.name, "parameters"))

    def __repr__(self):
        return f"GraphParameter: {self.name}"


def build_lazy_parameter(param, builder):
    model = builder.model
    param_keys = set_keys(param, builder)
    model._lazy_params.add(param)
    return LazyGraphParameter(param), param_keys


class LazyGraphParameter(ModelParameter):
    def __init__(self, param):
        self.param = param
        self._pkey = f"_lazy_{hash(param)}"

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return parameters[self._pkey]

    def __hash__(self):
        return hash(self.param)


class ComputedValueParameter(ModelParameter):
    def __init__(self, name):
        self.name = name

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return computed_values[self.name]

    def __hash__(self):
        return hash((self.name, "computed_values"))

    def __repr__(self):
        return f"ComputedValue: {self.name}"

    def is_time_varying(self):
        return True


def concretize_arg(arg: Any, builder: ModelBuilder) -> Tuple[Any, List[str]]:
    """Resolve abstract and lazy parameters,
    returning the resolved object, and any input parameter keys it references

    Args:
        arg (_type_): _description_
        builder (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(arg, AbstractParameter):
        pkey = builder.find_key_from_obj(arg)
        return Variable(pkey, "parameters"), [pkey]
    elif isinstance(arg, LazyParameter):
        lazy_p, pkeys = build_lazy_parameter(arg, builder)
        return Variable(lazy_p._pkey, "parameters"), pkeys
    elif is_var(arg, "parameters"):
        return arg, [arg.name]
    else:
        return arg, []


def concretize_function_args(
    func_obj: Function, builder: ModelBuilder
) -> Tuple[Function, List[str]]:
    args = []
    all_params = []
    for a in func_obj.args:
        arg, pkeys = concretize_arg(a, builder)
        args.append(arg)
        all_params += pkeys
    kwargs = {}
    for k, v in func_obj.kwargs.items():
        arg, pkeys = concretize_arg(v, builder)
        kwargs[k] = arg
        all_params += pkeys
    return Function(func_obj.func, args, kwargs), all_params


def build_graph_function(func: Function, builder: ModelBuilder) -> Tuple[ModelParameter, List[str]]:
    func, pkeys = concretize_function_args(func, builder)
    return GraphFunction(func), pkeys


class GraphFunction(ModelParameter):
    def __init__(self, func):
        self.func = func

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        sources = dict(
            computed_values=computed_values, parameters=parameters, model_variables={"time": time}
        )
        args, kwargs = build_args(self.func.args, self.func.kwargs, sources)
        return self.func.func(*args, **kwargs)

    def __hash__(self):
        return hash(self.func)

    def __repr__(self):
        return f"GraphFunction: {self.func}"

    def is_time_varying(self):
        has_time_var = Time in self.func.args or Time in self.func.kwargs.values()
        has_computed_value = any([isinstance(arg, ComputedValue) for arg in self.func.args])
        has_computed_value = has_computed_value or any(
            [isinstance(arg, ComputedValue) for arg in self.func.kwargs.values()]
        )

        return has_time_var or has_computed_value


class PyFunction(ModelParameter):
    def __init__(self, func):
        self.func = func

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.func(time, computed_values)

    def __hash__(self):
        return hash(self.func)

    def __repr__(self):
        return f"PyFunction: {self.func}"

    def is_time_varying(self):
        return True


def build_compound_parameter(param: Iterable, builder: ModelBuilder = None):
    param_keys = []
    subparams = []
    for sub_param in param:
        subp, subp_keys = get_modelparameter_from_param(sub_param, builder)
        subparams += [subp]
        param_keys += subp_keys
    return CompoundParameter(subparams), subp_keys


class CompoundParameter(ModelParameter):
    def __init__(self, subparams: Tuple[ModelParameter], builder):
        self.subparams = subparams

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        value = self.subparams[0].get_value(time, computed_values, parameters)
        for subp in self.subparams[1:]:
            value *= subp.get_value(time, computed_values, parameters)
        return value

    def __hash__(self):
        return hash(self.subparams)

    def __repr__(self):
        return f"CompoundParameter: {self.subparams}"

    def is_time_varying(self):
        return any([sp.is_time_varying() for sp in self.subparams])


def get_modelparameter_from_param(
    param, builder: ModelBuilder = None, allow_any=False
) -> Tuple[ModelParameter, List[str]]:
    """Create a ModelParameter subclass object that can be called inside a model loop
    (ie time, computed_values and parameters are available)
    These are mostly flow params and adjustments
    Return this object, as well as a list of keys of all parameters found during its
    construction

    Args:
        param: Anything that might resolve to a single float
        builder (ModelBuilder, optional): _description_. Defaults to None.

    Raises:
        TypeError: Raised if the input param cannot be resolved

    Returns:
        The resolved ModelParameter, along with any parameter keys encountered in the resolution
    """
    if isinstance(param, ModelParameter):
        # We've already transformed this parameter
        return param, []
    elif isinstance(param, AbstractParameter):
        if builder is not None:
            pkey = builder.find_key_from_obj(param)
            return GraphParameter(pkey), [pkey]
        else:
            raise Exception("AbstractParameter encountered with no ModelBuilder for lookup", param)
    elif isinstance(param, LazyParameter):
        if builder is not None:
            return build_lazy_parameter(param, builder)
        else:
            raise Exception(
                "Lazy AbstractParameter encountered with no ModelBuilder for lookup", param
            )
    elif is_var(param, "parameters"):
        return GraphParameter(param.name), [param.name]
    elif isinstance(param, Function):
        return build_graph_function(param, builder)
    elif isinstance(param, Real):
        return FloatParameter(param), []
    elif isinstance(param, list) or isinstance(param, tuple):
        return build_compound_parameter(param, builder)
    elif callable(param):
        return PyFunction(param), []
    elif isinstance(param, ComputedValue):
        return ComputedValueParameter(param.name), []
    else:
        if allow_any:
            return DataParameter(param), []
        else:
            raise TypeError(f"Unsupported model parameter type {type(param)}", param)


def replace_with_typed_params(m, builder):
    # Inside flows
    for f in m._flows:
        f.param = get_modelparameter_from_param(f.param, builder)

        for adj in f.adjustments:
            if adj is not None:
                adj.param = get_modelparameter_from_param(adj.param, builder)

    for k, v in m._derived_output_requests.items():
        req_type = v["request_type"]
        if req_type == "param_func":
            v["func"] = concretize_function_args(v["func"], builder)

    # Inside stratifications - we have retained some useful information...
    for s in m._stratifications:
        # Once a model is built, these become meaningless,
        # so it needs to happen at the start...
        # Keep this code here for now just in case we want to implement this in a different order

        # Flow adjustments live here quite happily
        # Flow _parameters_ however are stratified to oblivion, hence the section above ^^^^^
        for fname, adjustments in s.flow_adjustments.items():
            for adj, source_strata, dest_strata in adjustments:
                for k, v in adj.items():
                    if v is not None:
                        # Do nothing, reinstate if we reorder this
                        pass
                        # v.param = get_modelparameter_from_param(v.param)


def finalize_parameters(model, builder: ModelBuilder = None):
    """Called as part of model.finalize
    This ensures all parameters (and function calls) have concrete computegraph.Variable
    realisations.  We can scan all possible parameterized sites, and replace params
    with appropriate types, as well as collecting all referenced named parameters

    Args:
        builder: The ModelBuilder responsible for this model
    """

    all_params = []

    # Flow parameters and adjustments
    for f in model._flows:
        f.param, pkeys = get_modelparameter_from_param(f.param, builder)
        all_params += pkeys

        for adj in f.adjustments:
            if adj is not None:
                adj.param, pkeys = get_modelparameter_from_param(adj.param, builder)
                all_params += pkeys

    # Initial population
    # Can be one of
    # dict, ParamVariable (or AbstractParameter), or Function
    if not isinstance(model._initial_population_distribution, dict):
        param, pkeys = get_modelparameter_from_param(
            model._initial_population_distribution, builder
        )
        model._initial_population_distribution = param
        all_params += pkeys

    # Stratifications - population split, infectiousness adjustments, mixing matrix
    for s in model._stratifications:
        if not isinstance(s.population_split, dict):
            param, pkeys = get_modelparameter_from_param(s.population_split, builder)
            s.population_split = param
            all_params += pkeys

        for comp, adjustments in s.infectiousness_adjustments.items():
            for strain, adjustment in adjustments.items():
                if adjustment is not None:
                    param, pkeys = get_modelparameter_from_param(adjustment.param, builder)
                    adjustment.param = param
                    all_params += pkeys

        if s.mixing_matrix is not None:
            param, pkeys = get_modelparameter_from_param(s.mixing_matrix, builder, True)
            s.mixing_matrix = param
            all_params += pkeys

    # Computed values
    cv_graph = {}

    for k, v in model._computed_values_graph_dict.items():
        param, pkeys = concretize_function_args(v, builder)
        cv_graph[k] = param
        all_params += pkeys

    model._computed_values_graph_dict = cv_graph

    # Derived outputs
    for k, v in model._derived_output_requests.items():
        req_type = v["request_type"]
        if req_type == "param_func":
            param, pkeys = concretize_function_args(v["func"], builder)
            v["func"] = param
            all_params += pkeys

    return set(all_params)
