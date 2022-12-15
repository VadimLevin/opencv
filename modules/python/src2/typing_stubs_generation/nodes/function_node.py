from typing import NamedTuple, Sequence, Tuple, Type, Optional, List, Union

from .node import ASTNode


class FunctionNode(ASTNode):
    class Arg(NamedTuple):
        name: str
        typename: Optional[str] = None
        default_value: Optional[str] = None

    class RetType(NamedTuple):
        types: Union[str, Sequence[str]] = ()

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
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return ()

    def add_overload(self, arguments: Sequence["FunctionNode.Arg"] = (),
                     return_type: Optional["FunctionNode.RetType"] = None):
        self.overloads.append(FunctionNode.Overload(arguments, return_type))
