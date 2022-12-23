from __future__ import annotations

from typing import NamedTuple, Sequence, Type

from .node import ASTNode, ASTNodeType
from .type_node import TypeNode, NoneTypeNode, TypeResolutionError


class FunctionNode(ASTNode):
    class Arg(NamedTuple):
        name: str
        type_node: TypeNode | None = None
        default_value: str | None = None

        @property
        def typename(self) -> str | None:
            return getattr(self.type_node, "full_typename", None)

        def relative_typename(self, root: str) -> str | None:
            if self.type_node is not None:
                return self.type_node.relative_typename(root)
            return None

    class RetType(NamedTuple):
        type_node: TypeNode = NoneTypeNode("void")

        @property
        def typename(self) -> str:
            return self.type_node.full_typename

        def relative_typename(self, root: str) -> str | None:
            return self.type_node.relative_typename(root)

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
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Function

    @property
    def children_types(self) -> tuple[Type[ASTNode], ...]:
        return ()

    def add_overload(self, arguments: Sequence["FunctionNode.Arg"] = (),
                     return_type: "FunctionNode.RetType" | None = None):
        self.overloads.append(FunctionNode.Overload(arguments, return_type))

    def resolve_type_nodes(self, root: ASTNode):
        def has_unresolved_type_node(item) -> bool:
            return item.type_node is not None and not item.type_node.is_resolved

        errors = []
        for overload in self.overloads:
            for arg in filter(has_unresolved_type_node, overload.arguments):
                try:
                    arg.type_node.resolve(root)  # type: ignore
                except TypeResolutionError as e:
                    errors.append(
                        'Failed to resolve "{}" argument: {}'.format(arg.name, e)
                    )
            if overload.return_type is not None and \
                    has_unresolved_type_node(overload.return_type):
                try:
                    overload.return_type.type_node.resolve(root)
                except TypeResolutionError as e:
                    errors.append('Failed to resolve return type: {}'.format(e))
        if len(errors) > 0:
            raise TypeResolutionError(
                'Failed to resolve "{}" function against "{}". Errors: {}'.format(
                    self.full_export_name, root.full_export_name,
                    ", ".join("[{}]: {}".format(i, e) for i, e in enumerate(errors))
                )
            )
