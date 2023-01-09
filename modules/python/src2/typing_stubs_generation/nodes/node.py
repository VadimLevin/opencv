import abc
import enum
import itertools
from typing import (Iterator, Type, TypeVar, Iterable, Dict,
                    Optional, Tuple, DefaultDict)
from collections import defaultdict

import weakref


ASTNodeSubtype = TypeVar("ASTNodeSubtype", bound="ASTNode")
NodeType = Type["ASTNode"]
NameToNode = Dict[str, ASTNodeSubtype]


class ASTNodeType(enum.Enum):
    Namespace = enum.auto()
    Class = enum.auto()
    Function = enum.auto()
    Enumeration = enum.auto()
    Constant = enum.auto()


class ASTNode:
    def __init__(self, name: str, parent: Optional["ASTNode"] = None,
                 export_name: Optional[str] = None) -> None:
        FORBIDDEN_SYMBOLS = ";,*&#/|\\@!()[]^% "
        for forbidden_symbol in FORBIDDEN_SYMBOLS:
            assert forbidden_symbol not in name, \
                "Invalid node identifier '{}' - contains 1 or more "\
                "forbidden symbols: ({})".format(name, FORBIDDEN_SYMBOLS)

        assert ":" not in name, \
            "Name '{}' contains C++ scope symbols (':'). Convert the name to "\
            "Python style and create appropriate parent nodes".format(name)

        assert "." not in name, \
            "Trying to create a node with '.' symbols in its name ({}). " \
            "Dots are supposed to be a scope delimiters, so create all nodes in ('{}') " \
            "and add '{}' as a last child node".format(
                name,
                "->".join(name.split('.')[:-1]),
                name.rsplit('.', maxsplit=1)[-1]
            )

        self.__name = name
        self.export_name = name if export_name is None else export_name
        self._parent: Optional["ASTNode"] = None
        self.parent = parent
        self.is_exported = True
        self._children: DefaultDict[NodeType, NameToNode] = defaultdict(dict)

    def __str__(self) -> str:
        return "{}('{}' exported as '{}')".format(
            type(self).__name__.replace("Node", ""), self.name, self.export_name
        )

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractproperty
    def children_types(self) -> Tuple[Type["ASTNode"], ...]:
        pass

    @abc.abstractproperty
    def node_type(self) -> ASTNodeType:
        pass

    @property
    def dependencies(self) -> Iterable["ASTNode"]:
        return itertools.chain(*(node.dependencies
                               for children in self._children.values()
                               for node in children.values()))

    @property
    def name(self) -> str:
        return self.__name

    @property
    def parent(self) -> Optional["ASTNode"]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional["ASTNode"]) -> None:
        assert value is None or isinstance(value, ASTNode), \
            "ASTNode.parent should be None or another ASTNode, " \
            "but got: {}".format(type(value))

        if value is not None:
            value.__check_child_before_add(type(self), self.name)

        # Detach from previous parent
        if self._parent is not None:
            self._parent._children[type(self)].pop(self.name)

        if value is None:
            self._parent = None
            return

        # Set a weak reference to a new parent and add self to its children
        self._parent = weakref.proxy(value)
        value._children[type(self)][self.name] = self

    @property
    def native_name(self) -> str:
        return self.full_name.replace(".", "::")

    @property
    def full_name(self) -> str:
        return self._construct_full_name("name")

    @property
    def full_export_name(self) -> str:
        return self._construct_full_name("export_name")

    def __check_child_before_add(self, child_type: Type[ASTNodeSubtype],
                                 name: str) -> None:
        assert len(self.children_types) > 0, \
            "Trying to add child node '{}::{}' to node '{}::{}' " \
            "that can't have children nodes".format(child_type.__name__, name,
                                                    type(self).__name__,
                                                    self.name)

        assert child_type in self.children_types, \
            "Trying to add child node '{}::{}' to node '{}::{}' " \
            "that supports only ({}) as its children types".format(
                child_type.__name__, name, type(self).__name__, self.name,
                ",".join(t.__name__ for t in self.children_types)
            )

        if self._find_child(child_type, name) is not None:
            raise ValueError(
                "Node '{}::{}' already has a child '{}::{}'".format(
                    type(self).__name__, self.name, child_type.__name__, name
                )
            )

    def _add_child(self, child_type: Type[ASTNodeSubtype], name: str,
                   **kwargs) -> ASTNodeSubtype:
        self.__check_child_before_add(child_type, name)
        return child_type(name, parent=self, **kwargs)

    def _find_child(self, child_type: Type[ASTNodeSubtype],
                    name: str) -> Optional[ASTNodeSubtype]:
        if child_type not in self._children:
            return None
        return self._children[child_type].get(name, None)

    def _construct_full_name(self, property_name: str) -> str:
        def get_name(node: ASTNode) -> str:
            return getattr(node, property_name)

        name_parts = [get_name(self), ]
        parent = self.parent
        while parent is not None:
            name_parts.append(get_name(parent))
            parent = parent.parent
        return ".".join(reversed(name_parts))

    def __iter__(self) -> Iterator["ASTNode"]:
        return iter(itertools.chain.from_iterable(
            node
            # Iterate over mapping between node type and nodes dict
            for nodes in self._children.values()
            # Iterate over mapping between node name and node
            for node in nodes.values()
        ))
