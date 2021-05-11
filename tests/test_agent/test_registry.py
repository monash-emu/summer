"""
TODO:

    - test Registry.add_node
    - test Registry.remove_node
    - test Registry.get_nodes
    - test Registry.get_node_contacts
    - test Query.exclude
    - test Query.deselect
    - test Query.only
    - test Query.id

"""
import numpy as np
from numpy.testing import assert_array_equal

from summer.agent import registry
from summer.agent.registry import Registry, vectorized, arguments
from summer.agent.entities import BaseEntity
from summer.agent.fields import IntegerField


class DummyEntity(BaseEntity):
    age = IntegerField(default=30)
    weight = IntegerField(distribution=lambda: 70)


def test_regsitry_init__with_no_expected_all():
    reg = Registry(DummyEntity)
    reg.count == 0
    assert_array_equal(reg.vals["age"], np.array([]))
    assert_array_equal(reg.vals["weight"], np.array([]))


def test_regsitry_init__with_expected_all():
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


def test_regsitry_add():
    reg = _get_dummy_registry(3)
    reg.count == 3
    assert_array_equal(reg.vals["age"], np.array([1, 2, 3]))
    assert_array_equal(reg.vals["weight"], np.array([4, 5, 6]))

    reg.add(DummyEntity(age=7, weight=13))

    reg.count == 4
    assert_array_equal(reg.vals["age"], np.array([1, 2, 3, 7]))
    assert_array_equal(reg.vals["weight"], np.array([4, 5, 6, 13]))


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


def test_registry_query_all():
    reg = _get_dummy_registry(3)
    ents = reg.query.all()
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


def test_registry_query_select():
    reg = _get_dummy_registry(10)
    # Nothing provided
    assert reg.query.select([]).ids() == []
    # Ids are selected
    assert reg.query.select([0, 5, 8]).ids() == [0, 5, 8]
    # Only intersection of currently selected ids and requested selection are selected.
    assert reg.query.filter(age__gte=6).ids() == [5, 6, 7, 8, 9]
    assert reg.query.filter(age__gte=6).select([0, 5, 8]).ids() == [5, 8]


def test_registry_query_choose(monkeypatch):
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


def test_registry_query_where__with_scalar_arguments():
    reg = _get_dummy_registry(10)

    # Pick nothing
    assert reg.query.where(lambda: False).ids() == []
    # Pick everything
    assert reg.query.where(lambda: True).ids() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Pick every second item
    @arguments("id")
    def id_func(entity_id):
        assert type(entity_id) is np.int64
        return entity_id % 2 == 0

    assert reg.query.where(id_func).ids() == [0, 2, 4, 6, 8]

    # Pick items where age and weight
    @arguments("id")
    def id_func(entity_id):
        assert type(entity_id) is np.int64
        return entity_id % 2 == 0

    # weight:           11 12 13 14 15 16 17 18 19 20
    # age:              1  2  3  4  5  6  7  8  9  10
    # weight / age:     11 6  -  -  3  -  -  -  -  2
    # expected ids:     0  1  -  -  4  -  -  -  -  9
    # Pick items where some function age and weight is True.
    @arguments("age", "weight")
    def id_func(age, weight):
        assert type(age) is np.int64
        assert type(weight) is np.int64
        return (weight / age) % 1 == 0

    assert reg.query.where(id_func).ids() == [0, 1, 4, 9]


def test_registry_query_where__with_vectorized_arguments():
    reg = _get_dummy_registry(10)

    # Pick nothing
    assert reg.query.where(lambda: False).ids() == []
    # Pick everything
    assert reg.query.where(lambda: True).ids() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Pick every second item
    @vectorized
    @arguments("id")
    def id_func(entity_id):
        assert type(entity_id) is np.ndarray
        return entity_id % 2 == 0

    assert reg.query.where(id_func).ids() == [0, 2, 4, 6, 8]

    # Pick items where age and weight
    @vectorized
    @arguments("id")
    def id_func(entity_id):
        assert type(entity_id) is np.ndarray
        return entity_id % 2 == 0

    # weight:           11 12 13 14 15 16 17 18 19 20
    # age:              1  2  3  4  5  6  7  8  9  10
    # weight / age:     11 6  -  -  3  -  -  -  -  2
    # expected ids:     0  1  -  -  4  -  -  -  -  9
    # Pick items where some function age and weight is True.
    @vectorized
    @arguments("age", "weight")
    def id_func(age, weight):
        assert type(age) is np.ndarray
        assert type(weight) is np.ndarray
        return (weight / age) % 1 == 0

    assert reg.query.where(id_func).ids() == [0, 1, 4, 9]


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
        assert type(age) is np.int64
        return age + 1

    @arguments("weight", "age")
    def weight_func(weight, age):
        assert type(age) is np.int64
        assert type(age) is np.int64
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
        assert type(age) is np.ndarray
        return age + 1

    @vectorized
    @arguments("weight", "age")
    def weight_func(weight, age):
        assert type(weight) is np.ndarray
        assert type(age) is np.ndarray
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
