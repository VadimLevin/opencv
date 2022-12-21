from __future__ import annotations

from typing import Sequence, Generator
import abc


class TypeNode(abc.ABC):
    def __init__(self, ctype_name: str) -> None:
        self.ctype_name = ctype_name

    @abc.abstractproperty
    def typename(self) -> str:
        pass

    @property
    def required_imports(self) -> Generator[str, None, None]:
        yield from ()


class NoneTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "None"


class AnyTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "Any"

    @property
    def required_imports(self) -> Generator[str, None, None]:
        yield "from typing import Any"


class PrimitiveTypeNode(TypeNode):
    def __init__(self, ctype_name: str, typename: str | None = None) -> None:
        super().__init__(ctype_name)
        self._typename = typename if typename is not None else ctype_name

    @property
    def typename(self) -> str:
        return self._typename

    @classmethod
    def int_(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "int"
        return PrimitiveTypeNode(ctype_name, typename="int")

    @classmethod
    def float_(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "float"
        return PrimitiveTypeNode(ctype_name, typename="float")

    @classmethod
    def bool_(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "bool"
        return PrimitiveTypeNode(ctype_name, typename="bool")

    @classmethod
    def str_(cls, ctype_name: str | None = None):
        if ctype_name is None:
            ctype_name = "string"
        return PrimitiveTypeNode(ctype_name, "str")


class AliasRefTypeNode(TypeNode):
    def __init__(self, alias_ctype_name: str,
                 alias_export_name: str | None = None):
        super().__init__(alias_ctype_name)
        if alias_export_name is None:
            self.alias_export_name = alias_ctype_name
        else:
            self.alias_export_name = alias_export_name

    @property
    def typename(self) -> str:
        return self.alias_export_name


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

    @property
    def required_imports(self) -> Generator[str, None, None]:
        return self.value.required_imports

    @classmethod
    def int_(cls, ctype_name: str, export_name: str | None = None,
             comment: str | None = None):
        return cls(ctype_name, PrimitiveTypeNode.int_(), export_name, comment)

    @classmethod
    def float_(cls, ctype_name: str, export_name: str | None = None,
               comment: str | None = None):
        return cls(ctype_name, PrimitiveTypeNode.float_(), export_name, comment)

    @classmethod
    def array_(cls, ctype_name: str, shape: tuple[int, ...] | None,
               dtype: str | None = None, export_name: str | None = None,
               comment: str | None = None):
        return cls(ctype_name, NDArrayTypeNode(ctype_name, shape, dtype),
                   export_name, comment)

    @classmethod
    def union_(cls, ctype_name: str, items: tuple[TypeNode, ...],
               export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, UnionTypeNode(ctype_name, items),
                   export_name, comment)

    @classmethod
    def optional_(cls, ctype_name: str, item: TypeNode,
                  export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, OptionalTypeNode(item), export_name, comment)

    @classmethod
    def sequence_(cls, ctype_name: str, item: TypeNode,
                  export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, SequenceTypeNode(ctype_name, item),
                   export_name, comment)

    @classmethod
    def tuple_(cls, ctype_name: str, items: tuple[TypeNode, ...],
               export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, TupleTypeNode(ctype_name, items),
                   export_name, comment)

    @classmethod
    def class_(cls, ctype_name: str, class_name: str,
               export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, ClassTypeNode(class_name),
                   export_name, comment)

    @classmethod
    def callable_(cls, ctype_name: str, argument_type: TypeNode,
                  return_type: TypeNode = NoneTypeNode("void"),
                  export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name,
                   CallableTypeNode(ctype_name, argument_type, return_type),
                   export_name, comment)

    @classmethod
    def ref_(cls, ctype_name: str, alias_ctype_name: str,
             alias_export_name: str | None = None,
             export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name,
                   AliasRefTypeNode(alias_ctype_name, alias_export_name),
                   export_name, comment)

    @classmethod
    def dict_(cls, ctype_name: str, key_type: TypeNode, value_type: TypeNode,
              export_name: str | None = None, comment: str | None = None):
        return cls(ctype_name, DictTypeNode(ctype_name, key_type, value_type),
                   export_name, comment)


class NDArrayTypeNode(TypeNode):
    def __init__(self, ctype_name: str, shape: tuple[int, ...] | None = None,
                 dtype: str | None = None) -> None:
        super().__init__(ctype_name)
        self.shape = shape
        self.dtype = dtype

    @property
    def typename(self) -> str:
        return "numpy.ndarray[{shape}, numpy.dtype[{dtype}]]".format(
            shape=self.shape if self.shape is not None else "Any",
            dtype=self.dtype if self.dtype is not None else "numpy.generic"
        )

    @property
    def required_imports(self) -> Generator[str, None, None]:
        yield "import numpy"


class ClassTypeNode(TypeNode):
    def __init__(self, ctype_name: str, typename: str | None = None,
                 module_name: str | None = None) -> None:
        super().__init__(ctype_name)
        self._typename = typename if typename is not None else ctype_name
        self._module_name = module_name

    @property
    def typename(self) -> str:
        return self._typename

    @property
    def required_imports(self) -> Generator[str, None, None]:
        if self._module_name is None:
            yield from super().required_imports
        else:
            # assert self._module_name is not None, \
            #     "Can't find a module for class '{}' exported as '{}'".format(
            #         self.ctype_name, self.typename,
            #     )
            yield "from {} import {}".format(self._module_name, self.typename)


class CollectionTypeNode(TypeNode):
    def __init__(self, ctype_name: str, items: Sequence[TypeNode]) -> None:
        super().__init__(ctype_name)
        self.items = list(items)

    @property
    def typename(self) -> str:
        return self.type_format.format(self.types_separator.join(
            item.typename for item in self.items
        ))

    @abc.abstractproperty
    def type_format(self) -> str:
        pass

    @abc.abstractproperty
    def types_separator(self) -> str:
        pass

    @property
    def required_imports(self) -> Generator[str, None, None]:
        for item in self.items:
            yield from item.required_imports


class SequenceTypeNode(CollectionTypeNode):
    def __init__(self, ctype_name: str, item: TypeNode) -> None:
        super().__init__(ctype_name, (item, ))

    @property
    def type_format(self):
        return "Sequence[{}]"

    @property
    def types_separator(self):
        return ", "

    @property
    def required_imports(self) -> Generator[str, None, None]:
        yield "from typing import Sequence"
        yield from super().required_imports


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


class CallableTypeNode(CollectionTypeNode):
    def __init__(self, ctype_name: str, argument_type: TypeNode,
                 return_type: TypeNode = NoneTypeNode("void")) -> None:
        super().__init__(ctype_name, (argument_type, return_type))

    @property
    def argument_type(self):
        return self.items[0]

    @property
    def return_type(self):
        return self.items[1]

    @property
    def typename(self) -> str:
        return "Callable[[{}], {}]".format(self.argument_type.typename,
                                           self.return_type.typename)

    @property
    def type_format(self) -> str:
        return "Callable[[{}]"

    @property
    def types_separator(self) -> str:
        return "], "

    @property
    def required_imports(self) -> Generator[str, None, None]:
        yield "from typing import Callable"
        yield from super().required_imports


class DictTypeNode(CollectionTypeNode):
    def __init__(self, ctype_name: str, key_type: TypeNode,
                 value_type: TypeNode) -> None:
        super().__init__(ctype_name, (key_type, value_type))

    @property
    def key_type(self) -> TypeNode:
        return self.items[0]

    @property
    def value_type(self) -> TypeNode:
        return self.items[1]

    @property
    def type_format(self):
        return "dict[{}]"

    @property
    def types_separator(self):
        return ", "
