from typing import Tuple, Type

from .node import ASTNode


class FunctionNode(ASTNode):
    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return ()
