from typing import Type

from .node import ASTNode


class ConstantNode(ASTNode):
    def __init__(self, name: str, value: str,
                 parent: ASTNode | None = None,
                 export_name: str | None = None) -> None:
        super().__init__(name, parent, export_name)
        self.value = value

    @property
    def children_types(self) -> tuple[Type[ASTNode], ...]:
        return ()

    def __str__(self) -> str:
        return "Constant('{}' exported as '{}': {})".format(
            self.name, self.export_name, self.value
        )
