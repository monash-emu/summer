import pytest

from summer.agent.query import Query

from summer.agent.entities import BaseEntity
from summer.agent.fields import IntegerField


class TestEntity(BaseEntity):
    age = IntegerField(default=30)
    weight = IntegerField(default=70)


def test_entity_fields():
    acutal_list = list(TestEntity.get_fields())
    expected_list = [
        ("age", TestEntity.age),
        ("weight", TestEntity.weight),
    ]
    for acutal, expected in zip(acutal_list, expected_list):
        act_name, act_field = acutal
        exp_name, exp_field = expected
        assert act_name == exp_name
        assert act_field is exp_field
