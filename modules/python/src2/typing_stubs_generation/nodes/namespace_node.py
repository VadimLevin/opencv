from typing import Tuple, Type, Dict, Iterable, Sequence
import itertools
import weakref

from .node import ASTNode

from .class_node import ClassNode, ClassProperty
from .function_node import FunctionNode
from .enumeration_node import EnumerationNode
from .constant_node import ConstantNode


class NamespaceNode(ASTNode):
    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (NamespaceNode, ClassNode, FunctionNode,
                EnumerationNode, ConstantNode)

    @property
    def dependencies(self) -> Iterable[ASTNode]:
        return itertools.chain(*(node.dependencies for node in itertools.chain(
            self.classes.values(),
            self.functions.values()
        )))

    @property
    def namespaces(self) -> Dict[str, "NamespaceNode"]:
        return self._children[NamespaceNode]

    @property
    def classes(self) -> Dict[str, ClassNode]:
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

    def add_namespace(self, name: str) -> "NamespaceNode":
        return self._add_child(NamespaceNode, name)

    def add_class(self, name: str,
                  bases: Sequence["weakref.ProxyType[ClassNode]"] = (),
                  properties: Sequence[ClassProperty] = ()) -> "ClassNode":
        return self._add_child(ClassNode, name, bases=bases,
                               properties=properties)

    def add_function(self, name: str) -> FunctionNode:
        return self._add_child(FunctionNode, name)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)
