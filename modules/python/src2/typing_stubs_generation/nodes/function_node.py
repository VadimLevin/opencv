from typing import NamedTuple, Sequence, Type, Optional, Tuple, List

from .node import ASTNode, ASTNodeType
from .type_node import TypeNode, NoneTypeNode, TypeResolutionError


class FunctionNode(ASTNode):
    class Arg(NamedTuple):
        name: str
        type_node: Optional[TypeNode] = None
        default_value: Optional[str] = None

        @property
        def typename(self) -> Optional[str]:
            return getattr(self.type_node, "full_typename", None)

        def relative_typename(self, root: str) -> Optional[str]:
            if self.type_node is not None:
                return self.type_node.relative_typename(root)
            return None

    class RetType(NamedTuple):
        type_node: TypeNode = NoneTypeNode("void")

        @property
        def typename(self) -> str:
            return self.type_node.full_typename

        def relative_typename(self, root: str) -> Optional[str]:
            return self.type_node.relative_typename(root)

    class Overload(NamedTuple):
        arguments: Sequence["FunctionNode.Arg"] = ()
        return_type: Optional["FunctionNode.RetType"] = None

    def __init__(self, name: str,
                 arguments: Optional[Sequence["FunctionNode.Arg"]] = None,
                 return_type: Optional["FunctionNode.RetType"] = None,
                 is_static: bool = False,
                 is_classmethod: bool = False,
                 parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None) -> None:
        super().__init__(name, parent, export_name)
        self.overloads: List[FunctionNode.Overload] = []
        self.is_static = is_static
        self.is_classmethod = is_classmethod
        if arguments is not None:
            self.add_overload(arguments, return_type)

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Function

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return ()

    def add_overload(self, arguments: Sequence["FunctionNode.Arg"] = (),
                     return_type: Optional["FunctionNode.RetType"] = None):
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
