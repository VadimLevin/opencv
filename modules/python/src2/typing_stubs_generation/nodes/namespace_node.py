from typing import Type, Iterable, Sequence, Tuple, Optional, Dict
import itertools
import weakref

from .node import ASTNode, ASTNodeType

from .class_node import ClassNode, ClassProperty
from .function_node import FunctionNode
from .enumeration_node import EnumerationNode
from .constant_node import ConstantNode

from .type_node import TypeResolutionError


class NamespaceNode(ASTNode):
    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Namespace

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (NamespaceNode, ClassNode, FunctionNode,
                EnumerationNode, ConstantNode)

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

    def add_function(self, name: str, arguments: Sequence[FunctionNode.Arg] = (),
                     return_type: Optional[FunctionNode.RetType] = None) -> FunctionNode:
        return self._add_child(FunctionNode, name, arguments=arguments,
                               return_type=return_type)

    def add_enumeration(self, name: str) -> EnumerationNode:
        return self._add_child(EnumerationNode, name)

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)

    def resolve_type_nodes(self, root: Optional[ASTNode] = None) -> None:
        for child in itertools.chain(self.functions.values(),
                                     self.classes.values(),
                                     self.namespaces.values()):
            try:
                child.resolve_type_nodes(self)  # type: ignore
            except TypeResolutionError:
                if root is not None:
                    child.resolve_type_nodes(root)  # type: ignore
                else:
                    raise
