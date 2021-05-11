from abc import ABC
from typing import Dict

from .fields import BaseField, GraphField, IntegerField


class EntityMeta(type):
    def __new__(cls, name, bases, dct):
        """
        Annotate new Entity class with 'fields' attributes.
        """
        new_cls = super().__new__(cls, name, bases, dct)
        fields = EntityMeta._get_fields(dct)

        for base_cls in bases:
            base_fields = EntityMeta._get_fields(base_cls.__dict__)
            for field_name in base_fields.keys():
                msg = f"Cannot define field '{field_name}' on class '{name}' because it is already fined in parent class {base_cls}"
                assert field_name not in fields, msg

            fields = {**fields, **base_fields}

        new_cls.fields = fields
        return new_cls

    @staticmethod
    def _get_fields(dct: dict):
        fields = {}
        for name, field in dct.items():
            if issubclass(field.__class__, BaseField):
                fields[name] = field

        return fields


class BaseEntity(metaclass=EntityMeta):
    fields: Dict[str, BaseField]

    def __init__(self, **kwargs):
        for field_name, field in self.fields.items():
            field_val = kwargs.get(field_name)
            if not field_val and field.default is not None:
                field_val = field.default
            elif not field_val and field.distribution is not None:
                field_val = field.distribution()
            elif not field_val:
                raise ValueError(f"Field {field_name} does not have a value.")

            assert field_val is not None
            field.validate(field_val)
            setattr(self, field_name, field_val)

    @classmethod
    def assert_fieldname(cls, field_name):
        assert (
            field_name in cls.fields.keys()
        ), f"Could not find field {field_name} in entity {cls}."


class BaseAgent(BaseEntity):
    pass


class BaseNetwork(BaseEntity):
    graph = GraphField()
    size = IntegerField(default=0)
