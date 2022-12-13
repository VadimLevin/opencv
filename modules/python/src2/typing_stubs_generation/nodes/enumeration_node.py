from typing import Tuple, Type, Dict

from .node import ASTNode

from .constant_node import ConstantNode


class EnumerationNode(ASTNode):
    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return (ConstantNode, )

    @property
    def constants(self) -> Dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)
