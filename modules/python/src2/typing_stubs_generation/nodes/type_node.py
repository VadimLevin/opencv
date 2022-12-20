from __future__ import annotations

from typing import Sequence
import abc


class TypeNode(abc.ABC):
    def __init__(self, ctype_name: str) -> None:
        self.ctype_name = ctype_name

    @abc.abstractproperty
    def typename(self) -> str:
        pass


class NoneTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "None"


class AnyTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "Any"


class PrimitiveTypeNode(TypeNode):
    def __init__(self, ctype_name: str, typename: str | None = None) -> None:
        super().__init__(ctype_name)
        self._typename = typename if typename is not None else ctype_name

    @property
    def typename(self) -> str:
        return self._typename

    @classmethod
    def float(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "float"
        return PrimitiveTypeNode(ctype_name, typename="float")

    @classmethod
    def bool(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "bool"
        return PrimitiveTypeNode(ctype_name, typename="bool")

    @classmethod
    def int(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "int"
        return PrimitiveTypeNode(ctype_name, typename="int")

    @classmethod
    def string(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "string"
        return PrimitiveTypeNode(ctype_name, "str")


class AliasLinkTypeNode(TypeNode):
    def __init__(self, ctype_name: str, alias_name: str | None = None):
        super().__init__(ctype_name)
        self.alias_name = alias_name if alias_name is not None else ctype_name

    @property
    def typename(self) -> str:
        return self.alias_name


class DirectAliasTypeNode(TypeNode):
    def __init__(self, ctype_name: str, value: TypeNode) -> None:
        super().__init__(ctype_name)
        self.value = value

    @property
    def typename(self) -> str:
        return self.value.typename


class AliasTypeNode(TypeNode):
    def __init__(self, ctype_name: str, value: TypeNode,
                 export_name: str | None = None,
                 comment: str | None = None) -> None:
        super().__init__(ctype_name)
        self.value = value
        self._export_name = export_name
        self.comment = comment

    @property
    def typename(self) -> str:
        if self._export_name is not None:
            return self._export_name
        return self.ctype_name

    @classmethod
    def int(cls, ctype_name: str, export_name: str | None = None,
            comment: str | None = None):
        return cls(ctype_name, PrimitiveTypeNode.int(), export_name, comment)

    @classmethod
    def float(cls, ctype_name: str, export_name: str | None = None,
              comment: str | None = None):
        return cls(ctype_name, PrimitiveTypeNode.float(), export_name, comment)

    @classmethod
    def array(cls, ctype_name: str, shape: tuple[int] | None,
              dtype: str | None = None, export_name: str | None = None,
              comment: str | None = None):
        return cls(ctype_name, NDArrayTypeNode(shape, dtype), export_name,
                   comment)

    @classmethod
    def union(cls, ctype_name: str, items: tuple[TypeNode],
              export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, UnionTypeNode(ctype_name, items),
                   export_name, comment)

    @classmethod
    def sequence(cls, ctype_name: str, item: TypeNode,
                 export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, SequenceTypeNode(ctype_name, item),
                   export_name, comment)

    @classmethod
    def tuple(cls, ctype_name: str, items: tuple[TypeNode],
              export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, TupleTypeNode(ctype_name, items),
                   export_name, comment)

    @classmethod
    def class_(cls, ctype_name: str, class_name: str,
               export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, ClassTypeNode(class_name),
                   export_name, comment)

    @classmethod
    def callable(cls, ctype_name: str, argument_type: TypeNode,
                 return_type: TypeNode = NoneTypeNode("void"),
                 export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name,
                   CallableTypeNode(ctype_name, argument_type, return_type),
                   export_name, comment)


class NDArrayTypeNode(TypeNode):
    def __init__(self, ctype_name: str, shape: tuple[int] | None = None,
                 dtype: str | None = None) -> None:
        super().__init__(ctype_name)
        self.shape = shape
        self.dtype = dtype

    @property
    def typename(self) -> str:
        return "numpy.ndarray[{shape}, numpy.dtype=[{dtype}]]".format(
            shape=self.shape if self.shape is not None else "typing.Any",
            dtype=self.dtype if self.dtype is not None else "numpy.generic"
        )


class ClassTypeNode(TypeNode):
    def __init__(self, ctype_name: str, typename: str | None = None) -> None:
        super().__init__(ctype_name)
        self._typename = typename if typename is not None else ctype_name

    @property
    def typename(self) -> str:
        return self._typename


class CollectionTypeNode(TypeNode):
    def __init__(self, ctype_name: str, items: Sequence[TypeNode]) -> None:
        super().__init__(ctype_name)
        self.items = list(items)

    @property
    def typename(self) -> str:
        types_separator = self.types_separator
        try:
            return self.type_format.format(types_separator.join(
                item.typename for item in self.items
            ))
        except TypeError:
            print(self.ctype_name, type(self))
            for i, item in enumerate(self.items):
                print(i, item.typename)
            raise

    @abc.abstractproperty
    def type_format(self) -> str:
        pass

    @abc.abstractproperty
    def types_separator(self) -> str:
        pass


class SequenceTypeNode(CollectionTypeNode):
    def __init__(self, ctype_name: str, item: TypeNode) -> None:
        super().__init__(ctype_name, (item, ))

    @property
    def type_format(self):
        return "Sequence[{}]"

    @property
    def types_separator(self):
        return ", "


class TupleTypeNode(CollectionTypeNode):
    @property
    def type_format(self):
        return "tuple[{}]"

    @property
    def types_separator(self) -> str:
        return ", "


class UnionTypeNode(CollectionTypeNode):
    @property
    def type_format(self):
        return "{}"

    @property
    def types_separator(self):
        return " | "


class OptionalTypeNode(UnionTypeNode):
    def __init__(self, value: TypeNode) -> None:
        super().__init__(value.ctype_name, (value, NoneTypeNode(value.ctype_name)))


class CallableTypeNode(TypeNode):
    def __init__(self, ctype_name: str, argument_type: TypeNode,
                 return_type: TypeNode = NoneTypeNode("void")) -> None:
        super().__init__(ctype_name)
        self.argument_type = argument_type
        self.return_type = return_type

    @property
    def typename(self) -> str:
        return "Callable[[{}], {}]".format(self.argument_type.typename,
                                           self.return_type.typename)


class DictTypeNode(TypeNode):
    def __init__(self, ctype_name: str, key_type: TypeNode,
                 value_type: TypeNode) -> None:
        super().__init__(ctype_name)
        self.key_type = key_type
        self.value_type = value_type

    @property
    def typename(self) -> str:
        return "dict[{}, {}]".format(self.key_type.typename,
                                     self.value_type.typename)
