from __future__ import annotations

__all__ = ("generate_typing_stubs", "generate_aliases_module", )

from io import StringIO
from pathlib import Path
from typing import Generator, Type, Callable, NamedTuple

from .aliases import ALIASES

from .nodes import (ASTNode, NamespaceNode, ClassNode, FunctionNode,
                    EnumerationNode, ConstantNode)
from .nodes.type_node import (TypeNode, AliasTypeNode, AliasRefTypeNode,
                              CollectionTypeNode)


def generate_aliases_module(root: NamespaceNode, output_root: Path):
    def register_alias_links_from_collection(type_node: TypeNode):
        assert isinstance(type_node, CollectionTypeNode)
        for value_item in filter(lambda i: isinstance(i, AliasRefTypeNode),
                                 type_node.items):
            register_alias(ALIASES[value_item.ctype_name])  # type: ignore

    def register_alias(alias_node: AliasTypeNode):
        typename = alias_node.typename
        # Check if alias is already registered
        if typename in aliases:
            return
        if isinstance(alias_node.value, CollectionTypeNode):
            # Check if collection contains a link to another alias
            register_alias_links_from_collection(alias_node.value)

        aliases[typename] = alias_node.value.typename
        if alias_node.comment is not None:
            aliases[typename] += "  # " + alias_node.comment
        _add_required_imports(alias_node.value, required_imports)

    output_path = Path(output_root) / root.export_name / "typing"
    output_path.mkdir(parents=True, exist_ok=True)

    required_imports: set[str] = set()
    aliases: dict[str, str] = {}

    # For each node that is alias
    for node in filter(lambda a: isinstance(a, AliasTypeNode),
                       ALIASES.values()):
        register_alias(node)

    output_stream = StringIO()
    _write_required_imports(required_imports, output_stream)

    for alias_name, alias_type in aliases.items():
        output_stream.write(alias_name)
        output_stream.write(" = ")
        output_stream.write(alias_type)
        output_stream.write("\n")

    (output_path / "__init__.pyi").write_text(output_stream.getvalue())


def generate_typing_stubs(root: NamespaceNode, output_root: Path):
    output_path = Path(output_root) / root.export_name
    output_path.mkdir(parents=True, exist_ok=True)

    required_imports = _collect_required_imports(root)

    output_stream = StringIO()
    _write_required_imports(required_imports, output_stream)

    _generate_section_stub(StubSection("# Constants", ConstantNode), root,
                           output_stream, 0)
    # Special handling for enumerations...
    # Generate all enums from the module level
    has_enums = _generate_section_stub(StubSection("# Enumerations", EnumerationNode),
                                       root, output_stream, 0)
    # Collect all enums from class level and export them to module level
    for class_node in root.classes.values():
        if _generate_enums_from_classes_tree(class_node, output_stream, indent=0):
            has_enums = True
    # 2 empty lines between enum and classes definitions
    if has_enums:
        output_stream.write("\n")

    for section in STUB_SECTIONS:
        _generate_section_stub(section, root, output_stream, 0)
    (output_path / "__init__.pyi").write_text(output_stream.getvalue())
    for ns in root.namespaces.values():
        generate_typing_stubs(ns, output_path)


class StubSection(NamedTuple):
    name: str
    node_type: Type[ASTNode]


STUB_SECTIONS = (
    StubSection("# Constants", ConstantNode),
    # StubSection("# Enumerations", EnumerationNode), # Skipped for now (special rules)
    StubSection("# Classes", ClassNode),
    StubSection("# Functions", FunctionNode)
)


def _generate_section_stub(section, node, output_stream, indent):
    # type: (StubSection, ASTNode, StringIO, int) -> bool
    """Generates stub for a single type of children nodes of the provided node.

    Args:
        section (StubSection): section identifier that carries section name and
            type its nodes.
        node (ASTNode): root node with children nodes used for
        output_stream (StringIO): Output stream for section stub.
        indent (int): Indent used for each line output to `output_stream`.

    Returns:
        bool: `True` if section has a content, `False` otherwise.
    """
    if section.node_type not in node._children:
        return False

    children = node._children[section.node_type]
    if len(children) == 0:
        return False

    output_stream.write(" " * indent)
    output_stream.write(section.name)
    output_stream.write("\n")
    stub_generator = NODE_TYPE_TO_STUB_GENERATOR[section.node_type]
    for child in filter(lambda c: c.is_exported, children.values()):
        stub_generator(child, output_stream, indent)  # type: ignore
    output_stream.write("\n")
    return True


def _generate_class_stub(class_node, output_stream, indent=0):
    # type: (ClassNode, StringIO, int) -> None
    """Generates stub for the provided class node.

    Rules:
    - Read/write properties are converted to object attributes.
    - Readonly properties are converted to functions decorated with `@property`.
    - When return type of static functions matches class name - these functions
      are treated as factory functions and annotated with `@classmethod`.
    - In contrast to implicit `this` argument in C++ methods, in Python all
      "normal" methods have explicit `self` as their first argument.
    - Body of empty classes is replaced with `...`

    Example:
    ```cpp
    struct Object : public BaseObject {
        struct InnerObject {
            int param;
            bool param2;

            float readonlyParam();
        };

        Object(int param, bool param2 = false);

        Object(InnerObject obj);

        static Object create();

    };
    ```
    becomes
    ```python
    class Object(BaseObject):
        class InnerObject:
            param: int
            param2: bool

            @property
            def readonlyParam() -> float: ...

        @typing.override
        def __init__(self, param: int, param2: bool = ...) -> None: ...

        @typing.override
        def __init__(self, obj: "Object.InnerObject") -> None: ...

        @classmethod
        def create(cls) -> Object: ...
    ```

    Args:
        class_node (ClassNode): Class node to generate stub entry for.
        output_stream (StringIO): Output stream for class stub.
        indent (int, optional): Indent used for each line output to `output_stream`.
            Defaults to 0.
    """
    if len(class_node.bases) > 0:
        bases = "({})".format(
            ', '.join(base.export_name for base in class_node.bases))
    else:
        bases = ""

    output_stream.write(
        "{indent}class {name}{bases}:\n".format(
            indent=" " * indent,
            name=class_node.export_name,
            bases=bases
        )
    )
    has_content = len(class_node.properties) > 0

    # Processing class properties
    for property in class_node.properties:
        if property.is_readonly:
            template = "{indent}@property\n{indent}def {name}(self) -> {type}: ...\n"
        else:
            template = "{indent}{name}: {type}\n"

        output_stream.write(
            template.format(indent=" " * (indent + 4),
                            name=property.name,
                            type=property.typename)
        )
    if len(class_node.properties) > 0:
        output_stream.write("\n")

    for section in STUB_SECTIONS:
        if _generate_section_stub(section, class_node,
                                  output_stream, indent + 4):
            has_content = True
    if not has_content:
        output_stream.write(" " * (indent + 4))
        output_stream.write("...\n\n\n")


def _generate_constant_stub(constant_node: ConstantNode,
                            output_stream: StringIO, indent: int = 0,
                            extra_export_prefix: str = ""):
    output_stream.write(
        "{indent}{prefix}{name}: int\n".format(
            prefix=extra_export_prefix,
            name=constant_node.export_name,
            indent=" " * indent
        )
    )


def _generate_enumeration_stub(enumeration_node: EnumerationNode,
                               output_stream: StringIO, indent: int = 0,
                               extra_export_prefix: str = ""):
    """Generates stub for the provided enumeration node. In contrast to the
    Python `enum.Enum` class, C++ enumerations are exported as module-level
    (or class-level) constants.

    Example:
    ```cpp
    enum Flags {
        Flag1 = 0,
        Flag2 = 1,
        Flag3
    };
    ```
    becomes
    ```python
    Flag1: int
    Flag2: int
    Flag3: int
    Flags = int  # One of [Flag1, Flag2, Flag3]
    ```

    Unnamed enumerations don't export their names to Python:
    ```cpp
    enum {
        Flag1 = 0,
        Flag2 = 1
    };
    ```
    becomes
    ```python
    Flag1: int
    Flag2: int
    ```

    Scoped enumeration adds its name before each item name:
    ```cpp
    enum struct ScopedEnum {
        Flag1,
        Flag2
    };
    ```
    becomes
    ```python
    ScopedEnum_Flag1: int
    ScopedEnum_Flag2: int
    ScopedEnum = int # One of [ScopedEnum_Flag1, ScopedEnum_Flag2]
    ```

    Args:
        enumeration_node (EnumerationNode): Enumeration node to generate stub entry for.
        output_stream (StringIO): Output stream for enumeration stub.
        indent (int, optional): Indent used for each line output to `output_stream`.
            Defaults to 0.
    """

    entries_extra_prefix = extra_export_prefix
    if enumeration_node.is_scoped:
        entries_extra_prefix += enumeration_node.export_name + "_"
    for entry in enumeration_node.constants.values():
        _generate_constant_stub(entry, output_stream, indent, entries_extra_prefix)
    # Unnamed enumerations are skipped as definition
    if enumeration_node.export_name.endswith("<unnamed>"):
        output_stream.write("\n")
        return
    output_stream.write(
        "{indent}{export_prefix}{name} = int  # One of [{entries}]\n\n".format(
            export_prefix=extra_export_prefix,
            name=enumeration_node.export_name,
            entries=", ".join(entry.export_name
                              for entry in enumeration_node.constants.values()),
            indent=" " * indent
        )
    )


def _generate_function_stub(function_node: FunctionNode,
                            output_stream: StringIO, indent: int = 0):
    decorators = []
    if function_node.is_classmethod:
        decorators.append(" " * indent + "@classmethod")
    elif function_node.is_static:
        decorators.append(" " * indent + "@staticmethod")
    if len(function_node.overloads) > 1:
        decorators.append(" " * indent + "@typing.overload")
    for overload in function_node.overloads:
        # Annotate every function argument
        annotated_args = (arg.annotated_form for arg in overload.arguments)
        # And convert return type to the actual type
        ret_type = getattr(overload.return_type, "typename", "None")

        if function_node.parent.name == ret_type: # type: ignore
            ret_type = '"{}"'.format(ret_type)

        output_stream.write(
            "{decorators}"
            "{indent}def {name}({args}) -> {ret_type}: ...\n".format(
                decorators="\n".join(decorators) +
                "\n" if len(decorators) > 0 else "",
                name=function_node.export_name,
                args=", ".join(annotated_args),
                ret_type=ret_type,
                indent=" " * indent
            )
        )
    output_stream.write("\n")


def _generate_enums_from_classes_tree(class_node, output_stream,
                                      indent=0, class_name_prefix=""):
    # type: (ClassNode, StringIO, int, str) -> bool
    """Recursively generates class-level enumerations starting from the `class_node`.

    Args:
        class_node (ClassNode): _description_
        output_stream (StringIO): _description_
        indent (int, optional): _description_. Defaults to 0.
        class_name_prefix (str, optional): _description_. Defaults to "".

    Returns:
        bool: `True` if classes tree declares at least 1 enum, `False` otherwise.
    """
    class_name_prefix = class_node.export_name + "_" + class_name_prefix
    has_content = len(class_node.enumerations) > 0
    for enum_node in class_node.enumerations.values():
        _generate_enumeration_stub(enum_node, output_stream, indent,
                                   class_name_prefix)
    for cls in class_node.classes.values():
        if _generate_enums_from_classes_tree(cls, output_stream, indent,
                                             class_name_prefix):
            has_content = True
    return has_content


def check_overload_presence(node: NamespaceNode | ClassNode) -> bool:
    for func_node in node.functions.values():
        if len(func_node.overloads):
            return True
    return False


def _for_each_class(node: NamespaceNode | ClassNode) \
        -> Generator[ClassNode, None, None]:
    for cls in node.classes.values():
        yield cls
        if len(cls.classes):
            yield from _for_each_class(cls)


def _for_each_function(node: NamespaceNode | ClassNode) \
        -> Generator[FunctionNode, None, None]:
    for func in node.functions.values():
        yield func
    for cls in node.classes.values():
        yield from _for_each_function(cls)


def _for_each_function_overload(node: NamespaceNode | ClassNode) \
        -> Generator[FunctionNode.Overload, None, None]:
    for func in _for_each_function(node):
        for overload in func.overloads:
            yield overload


def _collect_required_imports(root: NamespaceNode) -> set[str]:
    required_imports: set[str] = set()
    # Check if typing module is required due to @overload decorator usage
    # Looking for module-level function with at least 1 overload
    has_overload = check_overload_presence(root)
    # if there is no module-level functions with overload, check its presence
    # during class traversing, including their inner-classes
    for cls in _for_each_class(root):
        if not has_overload and check_overload_presence(cls):
            has_overload = True
            required_imports.add("import typing")
        # Add required imports for class properties
        for prop in cls.properties:
            _add_required_imports(prop.type_node, required_imports)

    if has_overload:
        required_imports.add("import typing")
    # Importing external argument dependencies
    for overload in _for_each_function_overload(root):
        for arg in filter(lambda a: a.type_node is not None, overload.arguments):
            _add_required_imports(arg.type_node, required_imports)  # type: ignore
        if overload.return_type is not None:
            _add_required_imports(overload.return_type.type_node,
                                  required_imports)

    for dep in root.dependencies:
        dep_parent = dep.parent
        assert dep_parent is not None, \
            "Logic Error! '{}' parent is None".format(dep.name)

        # if dependency is not local add it to import list
        if dep_parent != root:
            required_import = "from {} import {}".format(
                dep_parent.full_export_name, dep.export_name
            )
            required_imports.add(required_import)
    return required_imports


def _add_required_imports(type_node: TypeNode, required_imports: set[str]):
    if isinstance(type_node, AliasTypeNode):
        required_import = "from cv2.typing import " + type_node.typename
        required_imports.add(required_import)
    else:
        for required_import in type_node.required_imports:
            required_imports.add(required_import)


def _write_required_imports(required_imports: set[str], output_stream: StringIO):
    for required_import in required_imports:
        output_stream.write(required_import)
        output_stream.write("\n")
    if len(required_imports):
        output_stream.write("\n\n")


StubGenerator = Callable[[ASTNode, StringIO, int], None]


NODE_TYPE_TO_STUB_GENERATOR = {
    ClassNode: _generate_class_stub,
    ConstantNode: _generate_constant_stub,
    EnumerationNode: _generate_enumeration_stub,
    FunctionNode: _generate_function_stub
}
