from __future__ import annotations

from typing import Sequence, Generator
import weakref
import abc

from .node import ASTNode, ASTNodeType


class TypeNode(abc.ABC):
    def __init__(self, ctype_name: str) -> None:
        self.ctype_name = ctype_name

    @abc.abstractproperty
    def typename(self) -> str:
        pass

    @property
    def full_typename(self) -> str:
        return self.typename

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield from ()

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield from ()

    def relative_typename(self, root: str) -> str:
        return self.full_typename

    def resolve(self, root: ASTNode):
        pass


class NoneTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "None"


class AnyTypeNode(TypeNode):
    @property
    def typename(self) -> str:
        return "typing.Any"

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import typing"


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

    @property
    def full_typename(self) -> str:
        return "cv2.typing." + self.typename


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
    def full_typename(self) -> str:
        return "cv2.typing." + self.typename

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        return self.value.required_usage_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import cv2.typing"

    def resolve(self, root: ASTNode):
        try:
            self.value.resolve(root)
        except Exception as e:
            raise ValueError(
                "Failed to resolve alias '{}' type exported as '{}'".format(
                    self.ctype_name, self.typename
                )
            ) from e

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
        if comment is None:
            comment = "Shape: " + str(shape)
        else:
            comment += ". Shape: " + str(shape)
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
            # NOTE: Shape is not fully supported yet
            # shape=self.shape if self.shape is not None else "typing.Any",
            shape="typing.Any",
            dtype=self.dtype if self.dtype is not None else "numpy.generic"
        )

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import numpy"
        # if self.shape is None:
        yield "import typing"


class ClassTypeNode(TypeNode):
    def __init__(self, ctype_name: str, typename: str | None = None,
                 module_name: str | None = None) -> None:
        super().__init__(ctype_name)
        self._typename = typename if typename is not None else ctype_name
        self._module_name = module_name
        self._ast_node: weakref.ProxyType[ASTNode] | None = None

    @property
    def typename(self) -> str:
        if self._ast_node is None:
            return self._typename
        typename = self._ast_node.export_name
        if self._ast_node.node_type is not ASTNodeType.Enumeration:
            return typename
        # NOTE: Special handling for enums
        parent = self._ast_node.parent
        while parent.node_type is ASTNodeType.Class:
            typename = parent.export_name + "_" + typename
            parent = parent.parent
        return typename

    @property
    def full_typename(self) -> str:
        if self._ast_node is not None:
            if self._ast_node.node_type is not ASTNodeType.Enumeration:
                return self._ast_node.full_export_name
            # NOTE: enumerations are exported to module scope
            typename = self._ast_node.export_name
            parent = self._ast_node.parent
            while parent.node_type is ASTNodeType.Class:
                typename = parent.export_name + "_" + typename
                parent = parent.parent
            return parent.full_export_name + "." + typename
        if self._module_name is not None:
            return self._module_name + "." + self._typename
        return self._typename

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        if self._module_name is None:
            assert self._ast_node is not None, \
                "Can't find a module for class '{}' exported as '{}'".format(
                    self.ctype_name, self.typename,
                )
            module = self._ast_node.parent
            while module.node_type is not ASTNodeType.Namespace:
                module = module.parent
            yield "import " + module.full_export_name
        else:
            yield "import " + self._module_name

    def resolve(self, root: ASTNode):
        # Symbol already resolved
        if self._ast_node is not None or self._module_name is not None:
            return

        node = _resolve_symbol(root, self.typename)
        if node is None:
            raise ValueError("Failed to resolve '{}' exposed as '{}'".format(
                self.ctype_name, self.typename
            ))
        self._ast_node = weakref.proxy(node)

    def relative_typename(self, root: str) -> str:
        assert self._ast_node is not None or self._module_name is not None, \
            "'{}' exported as '{}' is not resolved yet".format(self.ctype_name,
                                                               self.typename)
        if self._module_name is None:
            module = self._ast_node.parent  # type: ignore
            while module.node_type is not ASTNodeType.Namespace:
                module = module.parent
            module_name = module.full_export_name
        else:
            module_name = self._module_name
        if module_name != root:
            return self.full_typename
        return self.typename


class CollectionTypeNode(TypeNode):
    def __init__(self, ctype_name: str, items: Sequence[TypeNode]) -> None:
        super().__init__(ctype_name)
        self.items = list(items)

    @property
    def typename(self) -> str:
        return self.type_format.format(self.types_separator.join(
            item.typename for item in self.items
        ))

    @property
    def full_typename(self) -> str:
        return self.type_format.format(self.types_separator.join(
            item.full_typename for item in self.items
        ))

    def resolve(self, root: ASTNode):
        for item in self.items:
            item.resolve(root)

    def relative_typename(self, root: str) -> str:
        return self.type_format.format(self.types_separator.join(
            item.relative_typename(root) for item in self.items
        ))

    @abc.abstractproperty
    def type_format(self) -> str:
        pass

    @abc.abstractproperty
    def types_separator(self) -> str:
        pass

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        for item in self.items:
            yield from item.required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        for item in self.items:
            yield from item.required_usage_imports


class SequenceTypeNode(CollectionTypeNode):
    def __init__(self, ctype_name: str, item: TypeNode) -> None:
        super().__init__(ctype_name, (item, ))

    @property
    def type_format(self):
        return "typing.Sequence[{}]"

    @property
    def types_separator(self):
        return ", "

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield "import typing"
        yield from super().required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import typing"
        yield from super().required_usage_imports


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
    def type_format(self) -> str:
        return "typing.Callable[[{}]"

    @property
    def types_separator(self) -> str:
        return "], "

    @property
    def required_definition_imports(self) -> Generator[str, None, None]:
        yield "import typing"
        yield from super().required_definition_imports

    @property
    def required_usage_imports(self) -> Generator[str, None, None]:
        yield "import typing"
        yield from super().required_usage_imports


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


def _resolve_symbol(root: ASTNode | None, symbol: str) -> ASTNode | None:
    def search_down_symbol(scope: ASTNode | None,
                           scope_sep: str) -> ASTNode | None:
        parts = symbol.split(scope_sep, maxsplit=1)
        while len(parts) == 2:
            # Try to find narrow scope
            scope = _resolve_symbol(scope, parts[0])
            if scope is None:
                return None
            # and resolve symbol in it
            node = _resolve_symbol(scope, parts[1])
            if node is not None:
                return node
            # symbol is not found, but narrowed scope is valid - diving further
            parts = parts[1].split(scope_sep, maxsplit=1)
        return None

    assert root is not None, \
        "Can't resolve symbol '{}' from NONE root".format(symbol)
    # Looking for exact symbol match
    for attr in filter(lambda attr: hasattr(root, attr),
                       ("namespaces", "classes", "enumerations")):
        nodes_dict = getattr(root, attr)  # type: dict[str, ASTNode]
        node = nodes_dict.get(symbol, None)
        if node is not None:
            return node
    # Symbol is not found, looking for more fine-grained scope if possible
    for scope_sep in ("_", "."):
        node = search_down_symbol(root, scope_sep)
        if node is not None:
            return node
    return None
