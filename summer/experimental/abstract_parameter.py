from numbers import Real

from summer.jaxify import get_modules

# np = get_modules()["numpy"]

from jax import numpy as np

op_table = {
    "*": np.multiply,
    "/": np.divide,
    "+": np.add,
    "-": np.subtract,
    "log": np.log,
    "exp": np.exp,
}


def children(lazy_obj):
    if isinstance(lazy_obj, LazyBinOP):
        return (lazy_obj.lhs, lazy_obj.rhs)
    elif isinstance(lazy_obj, LazyOneSidedOp):
        return (lazy_obj.arg,)
    else:
        return ()


def set_keys(lazy_obj, builder):
    out_keys = []
    if isinstance(lazy_obj, AbstractParameter):
        obj_key = builder.find_key_from_obj(lazy_obj)
        lazy_obj._set_key(obj_key)
        out_keys = out_keys + [obj_key]
    else:
        for child in children(lazy_obj):
            out_keys = out_keys = set_keys(child, builder)
    return out_keys


def evaluate_lazy(obj, parameters):
    if isinstance(obj, LazyParameter):
        return obj.evaluate(parameters)
    else:
        return obj


class LazyParameter:
    def evaluate(self, parameters):
        raise NotImplementedError

    def __add__(self, other):
        return LazyBinOP("+", self, other)

    def __radd__(self, other):
        return LazyBinOP("+", other, self)

    def __mul__(self, other):
        return LazyBinOP("*", self, other)

    def __rmul__(self, other):
        return LazyBinOP("*", other, self)

    def __sub__(self, other):
        return LazyBinOP("-", self, other)

    def __rsub__(self, other):
        return LazyBinOP("-", other, self)

    def __truediv__(self, other):
        return LazyBinOP("/", self, other)

    def __rtruediv__(self, other):
        return LazyBinOP("/", other, self)

    def exp(self):
        return LazyOneSidedOp("exp", self)

    def log(self):
        return LazyOneSidedOp("log", self)


class LazyOneSidedOp(LazyParameter):
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg

    def evaluate(self, parameters):
        return op_table[self.op](evaluate_lazy(self.arg, parameters))

    def __hash__(self):
        return hash((self.op, self.arg))


class LazyBinOP(LazyParameter):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.lhs} {self.op} {self.rhs})"

    def evaluate(self, parameters):
        return op_table[self.op](
            evaluate_lazy(self.lhs, parameters), evaluate_lazy(self.rhs, parameters)
        )

    def __hash__(self):
        return hash((self.op, self.lhs, self.rhs))


class AbstractParameter(LazyParameter):
    def _set_key(self, param_key):
        self._param_key = param_key

    def evaluate(self, parameters):
        return parameters[self._param_key]
