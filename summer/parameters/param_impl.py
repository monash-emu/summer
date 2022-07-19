from summer.parameters.params import build_args, is_var, Function


class ModelParameter:
    def get_value(self, time: float, computed_values: dict, parameters: dict):
        raise NotImplementedError

    def __eq__(self, other):
        return hash(other) == hash(self)


class FloatParameter(ModelParameter):
    def __init__(self, value):
        self.value = value

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.value

    def __hash__(self):
        return hash(self.value)


class GraphParameter(ModelParameter):
    def __init__(self, name):
        self.name = name

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return parameters[self.name]

    def __hash__(self):
        return hash((self.name, "parameters"))


class ComputedValue(ModelParameter):
    def __init__(self, name):
        self.name = name

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return computed_values[self.name]

    def __hash__(self):
        return hash((self.name, "computed_values"))


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


class PyFunction(ModelParameter):
    def __init__(self, func):
        self.func = func

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        return self.func(time, computed_values)

    def __hash__(self):
        return hash(self.func)


class CompoundParameter(ModelParameter):
    def __init__(self, group):
        subparams = [get_modelparameter_from_param(p) for p in group]
        self.base_param = subparams[0]
        self.subparams = subparams[1:]

    def get_value(self, time: float, computed_values: dict, parameters: dict):
        value = self.base_param.get_value(time, computed_values, parameters)
        for subp in self.subparams:
            value *= subp.get_value(time, computed_values, parameters)
        return value


def get_modelparameter_from_param(param):
    if is_var(param, "parameters"):
        return GraphParameter(param.name)
    elif isinstance(param, Function):
        return GraphFunction(param)
    elif isinstance(param, float):
        return FloatParameter(param)
    elif isinstance(param, list) or isinstance(param, tuple):
        return CompoundParameter(param)
    elif callable(param):
        return PyFunction(param)
    elif isinstance(param, ModelParameter):
        # We've already updated this parameter
        return param
    else:
        raise Exception("", param)


def replace_with_typed_params(m):
    # Inside flows
    for f in m._flows:
        f.param = get_modelparameter_from_param(f.param)

        for adj in f.adjustments:
            if adj is not None:
                adj.param = get_modelparameter_from_param(adj.param)

    # Inside stratifications - we have retained some useful information...
    for s in m._stratifications:
        # Once a model is built, these become meaningless,
        # so it needs to happen at the start...

        # Flow adjustments live here quite happily
        # Flow _parameters_ however are stratified to oblivion, hence the section above ^^^^^
        for fname, adjustments in s.flow_adjustments.items():
            for adj, source_strata, dest_strata in adjustments:
                for k, v in adj.items():
                    if v is not None:
                        v.param = get_modelparameter_from_param(v.param)
