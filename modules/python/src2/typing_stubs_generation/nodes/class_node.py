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
    def __init__(self, name: str, parent: Optional["ASTNode"] = None,
                 export_name: Optional[str] = None,
                 bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                 properties: Sequence[ClassProperty] = ()) -> None:
        super().__init__(name, parent, export_name)
        self.bases = list(bases)
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
                  bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                  properties: Sequence[ClassProperty] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               properties=properties)

    def add_function(self, name: str, arguments: Sequence[FunctionNode.Arg] = (),
                     return_type: Optional[FunctionNode.RetType] = None,
                     is_static: bool = False) -> FunctionNode:
        arguments = list(arguments)
        if return_type is not None and isinstance(return_type.types, str):
            is_classmethod = return_type.types == self.name
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
