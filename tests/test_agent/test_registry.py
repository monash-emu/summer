import numpy as np
from numpy.testing import assert_array_equal

from summer.agent import registry
from summer.agent.registry import Registry, vectorized, arguments
from summer.agent.entities import BaseEntity
from summer.agent.fields import IntegerField


class DummyEntity(BaseEntity):
    age = IntegerField(default=30)
    weight = IntegerField(distribution=lambda: 70)


def test_regsitry_init__with_no_expected_entities():
    reg = Registry(DummyEntity)
    reg.count == 0
    assert_array_equal(reg.vals["age"], np.array([]))
    assert_array_equal(reg.vals["weight"], np.array([]))


def test_regsitry_init__with_expected_entities():
    reg = Registry(DummyEntity, 3)
    reg.count == 3
    assert_array_equal(reg.vals["age"], np.array([30, 30, 30]))
    assert_array_equal(reg.vals["weight"], np.array([70, 70, 70]))


def _get_dummy_registry(n):
    reg = Registry(DummyEntity, n)
    reg.vals["age"] = np.array(range(1, n + 1))
    reg.vals["weight"] = np.array(range(n + 1, 2 * n + 1))
    return reg


def test_registry_get():
    reg = _get_dummy_registry(3)
    assert reg.get(0).id == 0 and reg.get(0).age == 1 and reg.get(0).weight == 4
    assert reg.get(1).id == 1 and reg.get(1).age == 2 and reg.get(1).weight == 5
    assert reg.get(2).id == 2 and reg.get(2).age == 3 and reg.get(2).weight == 6


def test_registry_write():
    reg = _get_dummy_registry(3)
    reg.write(0, "age", 31)
    assert reg.get(0).id == 0 and reg.get(0).age == 31 and reg.get(0).weight == 4


def test_registry_save():
    reg = _get_dummy_registry(3)
    dummy_0, dummy_2 = reg.get(0), reg.get(2)
    dummy_0.age = 7
    dummy_2.weight = 23
    reg.save([dummy_0, dummy_2])
    assert_array_equal(reg.vals["age"], np.array([7, 2, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 5, 23]))


def test_registry_query_ids():
    reg = _get_dummy_registry(3)
    assert reg.query.ids() == [0, 1, 2]


def test_registry_query_entities():
    reg = _get_dummy_registry(3)
    ents = reg.query.entities()
    assert ents[0].id == 0 and ents[0].age == 1 and ents[0].weight == 4
    assert ents[1].id == 1 and ents[1].age == 2 and ents[1].weight == 5
    assert ents[2].id == 2 and ents[2].age == 3 and ents[2].weight == 6


def test_registry_query_filter():
    reg = _get_dummy_registry(3)
    # No args
    assert reg.query.filter().ids() == []
    # Equality
    assert reg.query.filter(age=1).ids() == [0]
    assert reg.query.filter(age=2).ids() == [1]
    assert reg.query.filter(age=3).ids() == [2]
    assert reg.query.filter(weight=5).ids() == [1]
    assert reg.query.filter(weight=5, age=2).ids() == [1]
    assert reg.query.filter(weight=5).filter(age=2).ids() == [1]
    assert reg.query.filter(weight=5, age=1).ids() == []
    assert reg.query.filter(weight=5).filter(age=1).ids() == []
    # Greater than
    assert reg.query.filter(age__gt=1).ids() == [1, 2]
    assert reg.query.filter(weight__gt=1).ids() == [0, 1, 2]
    assert reg.query.filter(age__gt=1).filter(weight__gt=5).ids() == [2]
    # Less than
    assert reg.query.filter(age__lt=2).ids() == [0]
    # Greater than or equal
    assert reg.query.filter(age__gte=2).ids() == [1, 2]
    # Less than or equal
    assert reg.query.filter(age__lte=2).ids() == [0, 1]
    # Not equal
    assert reg.query.filter(age__ne=2).ids() == [0, 2]


def test_registry_choose(monkeypatch):
    # Random version
    reg = _get_dummy_registry(3)
    assert len(reg.query.choose(0).ids()) == 0
    assert len(reg.query.choose(1).ids()) == 1
    assert len(reg.query.choose(2).ids()) == 2
    assert len(reg.query.choose(3).ids()) == 3

    # Deterministic version: always chooses 1st N.
    choices = lambda items, k: items[0:k]
    monkeypatch.setattr(registry.random, "choices", choices)

    assert reg.query.choose(0).ids() == []
    assert reg.query.choose(1).ids() == [0]
    assert reg.query.choose(2).ids() == [0, 1]
    assert reg.query.choose(3).ids() == [0, 1, 2]


def test_registry_query_update__with_static_values():
    reg = _get_dummy_registry(3)
    reg.query.filter(age=2).update(age=0, weight=100)
    assert_array_equal(reg.vals["age"], np.array([1, 0, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 100, 6]))

    reg = _get_dummy_registry(3)
    reg.query.filter(age__gte=2).update(age=0, weight=100)
    assert_array_equal(reg.vals["age"], np.array([1, 0, 0]))
    assert_array_equal(reg.vals["weight"], np.array([4, 100, 100]))


def test_registry_query_update__with_basic_function_values():
    reg = _get_dummy_registry(3)
    age_func, weight_func = lambda: 0, lambda: 100
    reg.query.filter(age=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 0, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 100, 6]))

    reg = _get_dummy_registry(3)
    reg.query.filter(age__gte=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 0, 0]))
    assert_array_equal(reg.vals["weight"], np.array([4, 100, 100]))


def test_registry_query_update__with_basic_function_values__with_arguments():
    reg = _get_dummy_registry(3)

    @arguments("age")
    def age_func(age):
        return age + 1

    @arguments("weight", "age")
    def weight_func(weight, age):
        return weight + age

    assert age_func._arg_names == ("age",)
    assert weight_func._arg_names == ("weight", "age")

    reg.query.filter(age=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 3, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 8, 6]))

    reg = _get_dummy_registry(3)
    reg.query.filter(age__gte=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 3, 4]))
    assert_array_equal(reg.vals["weight"], np.array([4, 8, 10]))


def test_registry_query_update__with_basic_function_values__with_vectorized_arguments():
    reg = _get_dummy_registry(3)

    @vectorized
    @arguments("age")
    def age_func(age):
        return age + 1

    @vectorized
    @arguments("weight", "age")
    def weight_func(weight, age):
        return weight + age

    assert age_func._arg_names == ("age",)
    assert age_func._vectorized
    assert weight_func._arg_names == ("weight", "age")
    assert weight_func._vectorized

    reg.query.filter(age=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 3, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 8, 6]))

    reg = _get_dummy_registry(3)
    reg.query.filter(age__gte=2).update(age=age_func, weight=weight_func)
    assert_array_equal(reg.vals["age"], np.array([1, 3, 4]))
    assert_array_equal(reg.vals["weight"], np.array([4, 8, 10]))
