from __future__ import annotations

from typing import NamedTuple, Sequence, Type

from .node import ASTNode
from .type_node import TypeNode, NoneTypeNode


class FunctionNode(ASTNode):
    class Arg(NamedTuple):
        name: str
        type_node: TypeNode | None = None
        default_value: str | None = None

        @property
        def annotated_form(self) -> str:
            annotated = self.name
            typename = self.typename
            if typename is not None:
                annotated += ": "
                annotated += typename
            if self.default_value is not None:
                annotated += " = ..."
            return annotated

        @property
        def typename(self) -> str | None:
            return getattr(self.type_node, "typename", None)

    class RetType(NamedTuple):
        type_node: TypeNode = NoneTypeNode("void")

        @property
        def typename(self) -> str:
            return self.type_node.typename

    class Overload(NamedTuple):
        arguments: Sequence["FunctionNode.Arg"] = ()
        return_type: "FunctionNode.RetType" | None = None

    def __init__(self, name: str,
                 arguments: Sequence["FunctionNode.Arg"] | None = None,
                 return_type: "FunctionNode.RetType" | None = None,
                 is_static: bool = False,
                 is_classmethod: bool = False,
                 parent: ASTNode | None = None,
                 export_name: str | None = None) -> None:
        super().__init__(name, parent, export_name)
        self.overloads: list[FunctionNode.Overload] = []
        self.is_static = is_static
        self.is_classmethod = is_classmethod
        if arguments is not None:
            self.add_overload(arguments, return_type)

    @property
    def children_types(self) -> tuple[Type[ASTNode], ...]:
        return ()

    def add_overload(self, arguments: Sequence["FunctionNode.Arg"] = (),
                     return_type: "FunctionNode.RetType" | None = None):
        self.overloads.append(FunctionNode.Overload(arguments, return_type))
