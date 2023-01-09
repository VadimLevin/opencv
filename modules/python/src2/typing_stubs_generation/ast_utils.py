from typing import NamedTuple, Sequence, Tuple, Union, List

from .nodes import NamespaceNode, ClassNode


class ScopeNotFoundError(Exception):
    pass


class SymbolName(NamedTuple):
    namespaces: Tuple[str, ...]
    classes: Tuple[str, ...]
    name: str

    def __str__(self) -> str:
        return "(namespace='{}', class='{}', name={})".format(
            '::'.join(self.namespaces),
            '::'.join(self.classes),
            self.name
        )

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, full_symbol_name: str,
              known_namespaces: Sequence[str]) -> "SymbolName":
        chunks = full_symbol_name.split('.')
        namespaces, name = chunks[:-1], chunks[-1]
        classes: List[str] = []
        while len(namespaces) > 0 and '.'.join(namespaces) not in known_namespaces:
            classes.insert(0, namespaces.pop())
        return SymbolName(tuple(namespaces), tuple(classes), name)


def find_scope(root: NamespaceNode, symbol_name: SymbolName,
               create_missing_namespaces: bool = True) -> Union[NamespaceNode, ClassNode]:
    assert symbol_name.namespaces[0] == root.name, \
        "Trying to find scope for '{}' with root namespace different from: '{}'".format(
            symbol_name, root.name
    )

    scope: Union[NamespaceNode, ClassNode] = root
    for namespace in symbol_name.namespaces[1:]:
        if namespace not in scope.namespaces:  # type: ignore
            if not create_missing_namespaces:
                raise ScopeNotFoundError(
                    "Can't find a scope for '{}', with '{}', because namespace"
                    " '{}' is not created yet and `create_missing_namespaces`"
                    " flag is set to False".format(
                        symbol_name.name, symbol_name, namespace
                    )
                )
            scope = scope.add_namespace(namespace)  # type: ignore
        else:
            scope = scope.namespaces[namespace]  # type: ignore
    for class_name in symbol_name.classes:
        if class_name not in scope.classes:
            raise ScopeNotFoundError(
                "Can't find a scope for '{}', with '{}', because '{}' "
                "class is not registered yet".format(
                    symbol_name.name, symbol_name, class_name
                )
            )
        scope = scope.classes[class_name]
    return scope
