from typing import Tuple, Type, Dict, Optional, Iterable, cast, List, Sequence, NamedTuple
import itertools

import weakref

from .node import ASTNode

from .function_node import FunctionNode
from .enumeration_node import EnumerationNode
from .constant_node import ConstantNode


class ClassProperty(NamedTuple):
    name: str
    typename: str
    is_readonly: bool


class ClassNode(ASTNode):
    def __init__(self, name: str, bases: Tuple["weakref.ProxyType[ClassNode]", ...] = (),
                 modifiers: Sequence[str] = (),
                 properties: Tuple[ClassProperty, ...] = (),
                 parent: Optional["ASTNode"] = None,
                 export_name: Optional[str] = None) -> None:
        super().__init__(name, parent, export_name)
        self.bases = bases
        self.modifiers: List[str] = list(modifiers)
        self.properties = properties
        self.__derived: List["weakref.ProxyType[ClassNode]"] = []
        for base in self.bases:
            base.add_derived_class(self)

    @property
    def weight(self) -> int:
        return -1 - sum(derived.weight for derived in self.__derived)

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (ClassNode, FunctionNode, EnumerationNode, ConstantNode)

    @property
    def dependencies(self) -> Iterable[ASTNode]:
        return itertools.chain(
            cast(Iterable[ASTNode], self.bases),
            *map(lambda func: func.dependencies, self.functions.values())
        )

    @property
    def classes(self) -> Dict[str, "ClassNode"]:
        return self._children[ClassNode]

    @property
    def functions(self) -> Dict[str, FunctionNode]:
        return self._children[FunctionNode]

    @property
    def enumerations(self) -> Dict[str, EnumerationNode]:
        return self._children[EnumerationNode]

    @property
    def constants(self) -> Dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_class(self, name: str,
                  bases: Tuple[weakref.ProxyType, ...] = (),
                  modifiers: Sequence[str] = (),
                  properties: Sequence[str] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               modifiers=modifiers,
                               properties=properties)

    def add_function(self, name: str) -> FunctionNode:
        return self._add_child(FunctionNode, name)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)

    def add_derived_class(self, derived_class_node: "ClassNode"):
        self.__derived.append(weakref.proxy(derived_class_node))
