from __future__ import annotations

from typing import Type

from .node import ASTNode

from .constant_node import ConstantNode


class EnumerationNode(ASTNode):
    def __init__(self, name: str, is_scoped: bool = False,
                 parent: ASTNode | None = None,
                 export_name: str | None = None) -> None:
        super().__init__(name, parent, export_name)
        self.is_scoped = is_scoped

    @property
    def children_types(self) -> tuple[Type[ASTNode], ...]:
        return (ConstantNode, )

    @property
    def constants(self) -> dict[str, ConstantNode]:
        return self._children[ConstantNode]

    def add_constant(self, name: str, value: str) -> ConstantNode:
        return self._add_child(ConstantNode, name, value=value)
