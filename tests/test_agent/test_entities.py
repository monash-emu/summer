import pytest

from summer.agent.entities import BaseEntity
from summer.agent.fields import IntegerField


class DummyEntity(BaseEntity):
    age = IntegerField(default=30)
    weight = IntegerField(default=70)


class DummyEntityTwo(BaseEntity):
    age = IntegerField(distribution=lambda: 31)
    weight = IntegerField(distribution=lambda: 71)


def test_entity_fields():
    """
    Expect get fields to return any BaseField attributes of the class.
    """
    assert DummyEntity.fields == {
        "age": DummyEntity.age,
        "weight": DummyEntity.weight,
    }


def test_entity__construction__with_defaults():
    """
    Expect constructing an entity instance to override fields
    with their default value.
    """
    test_entity = DummyEntity()
    assert test_entity.age == 30
    assert test_entity.weight == 70


def test_entity__construction__with_distribution():
    """
    Expect constructing an entity instance to override fields
    with their distribution value.
    """
    test_entity = DummyEntityTwo()
    assert test_entity.age == 31
    assert test_entity.weight == 71


def test_entity__construction__with_arguments():
    """
    Expect constructing an entity instance to override fields
    with their passed in value.
    """
    test_entity = DummyEntity(age=25, weight=50)
    assert test_entity.age == 25
    assert test_entity.weight == 50

    test_entity_two = DummyEntityTwo(age=25, weight=50)
    assert test_entity_two.age == 25
    assert test_entity_two.weight == 50


def test_entity__construction__with_arguments_validated():
    """
    Expect constructing an entity instance with the wrong type args
    to raise an error.
    """
    DummyEntity(age=25)  # An int.
    with pytest.raises(AssertionError):
        DummyEntity(age=25.2)  # Not an int.

    with pytest.raises(AssertionError):
        DummyEntity(age="25.2")  # Not an int.
