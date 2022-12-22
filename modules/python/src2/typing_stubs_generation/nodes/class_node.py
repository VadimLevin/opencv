from __future__ import annotations

from typing import Type, Iterable, cast, Sequence, NamedTuple
import itertools

import weakref

from .node import ASTNode, ASTNodeType

from .function_node import FunctionNode
from .enumeration_node import EnumerationNode
from .constant_node import ConstantNode

from .type_node import TypeNode


class ClassProperty(NamedTuple):
    name: str
    type_node: TypeNode
    is_readonly: bool

    @property
    def typename(self) -> str:
        return self.type_node.full_typename

    def resolve_type_nodes(self, root: ASTNode):
        try:
            self.type_node.resolve(root)
        except Exception as e:
            raise ValueError(
                "Failed to resolve '{}' property".format(self.name)
            ) from e

    def relative_typename(self, root: str) -> str | None:
        return self.type_node.relative_typename(root)


class ClassNode(ASTNode):
    def __init__(self, name: str, parent: ASTNode | None = None,
                 export_name: str | None = None,
                 bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                 properties: Sequence[ClassProperty] = ()) -> None:
        super().__init__(name, parent, export_name)
        self.bases = list(bases)
        self.properties = properties
        self.__derived: list["weakref.ProxyType[ClassNode]"] = []
        for base in self.bases:
            base.add_derived_class(self)

    @property
    def weight(self) -> int:
        return -1 - sum(derived.weight for derived in self.__derived)

    @property
    def children_types(self) -> tuple[Type[ASTNode], ...]:
        return (ClassNode, FunctionNode, EnumerationNode, ConstantNode)

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Class

    @property
    def dependencies(self) -> Iterable[ASTNode]:
        return itertools.chain(
            cast(Iterable[ASTNode], self.bases),
            *map(lambda func: func.dependencies, self.functions.values())
        )

    @property
    def classes(self) -> dict[str, "ClassNode"]:
        return self._children[ClassNode]

    @property
    def functions(self) -> dict[str, FunctionNode]:
        return self._children[FunctionNode]

    @property
    def enumerations(self) -> dict[str, EnumerationNode]:
        return self._children[EnumerationNode]

    @property
    def constants(self) -> dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_class(self, name: str,
                  bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                  properties: Sequence[ClassProperty] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               properties=properties)

    def add_function(self, name: str, arguments: Sequence[FunctionNode.Arg] = (),
                     return_type: FunctionNode.RetType | None = None,
                     is_static: bool = False) -> FunctionNode:
        arguments = list(arguments)
        if return_type is not None:
            is_classmethod = return_type.typename == self.name
        else:
            is_classmethod = False
        if not is_static:
            arguments.insert(0, FunctionNode.Arg("self"))
        elif is_classmethod:
            is_static = False
            arguments.insert(0, FunctionNode.Arg("cls"))
        return self._add_child(FunctionNode, name, arguments=arguments,
                               return_type=return_type, is_static=is_static,
                               is_classmethod=is_classmethod)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)

    def add_base(self, base_class_node: "ClassNode"):
        self.bases.append(weakref.proxy(base_class_node))
        base_class_node.add_derived_class(self)

    def add_derived_class(self, derived_class_node: "ClassNode"):
        self.__derived.append(weakref.proxy(derived_class_node))

    def resolve_type_nodes(self, root: ASTNode):
        errors = []
        for child in itertools.chain(self.functions.values(),
                                     self.classes.values(),
                                     self.properties):
            try:
                try:
                    # Give priority to narrowest scope (class-level scope in this case)
                    child.resolve_type_nodes(self)  # type: ignore
                except ValueError:
                    child.resolve_type_nodes(root)  # type: ignore
            except ValueError as e:
                errors.append(str(e))
        if len(errors) > 0:
            raise ValueError(
                "Failed to resolve '{}' class type nodes. Errors: {}".format(
                    self.full_export_name, errors
                )
            )


