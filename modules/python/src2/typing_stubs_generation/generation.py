__all__ = ("generate_typing_stubs", )

from io import StringIO
from pathlib import Path
from typing import Dict, Type, Callable, NamedTuple, Set

from .nodes import (ASTNode, NamespaceNode, ClassNode, FunctionNode,
                    EnumerationNode, ConstantNode)


def generate_typing_stubs(root: NamespaceNode, output_root: Path):
    output_path = Path(output_root) / root.export_name
    output_path.mkdir(parents=True, exist_ok=True)

    output_stream = StringIO()

    imported_dependencies: Set[str] = set()
    for dep in root.dependencies:
        dep_parent = dep.parent
        assert dep_parent is not None, \
            "Logic Error! '{}' parent is None".format(dep.name)

        # if dependency is not local add it to import list
        if dep_parent != root and dep.full_export_name not in imported_dependencies:
            imported_dependencies.add(dep.full_export_name)
            output_stream.write("from {} import {}\n".format(
                dep_parent.full_export_name, dep.export_name)
            )
    if len(imported_dependencies) > 0:
        output_stream.write("\n\n")

    _generate_section_stub(StubSection("# Constants", ConstantNode), root,
                           output_stream, 0)
    # Special handling for enumerations...
    # Generate all enums from the module level
    _generate_section_stub(StubSection("# Enumerations", EnumerationNode), root,
                           output_stream, 0)
    # Collect all enums from class level and export them to module level
    for class_node in root.classes.values():
        _generate_enums_from_classes_tree(class_node, output_stream, indent=0)

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


def _generate_section_stub(section: StubSection, node: ASTNode,
                           output_stream: StringIO, indent: int) -> bool:
    if section.node_type not in node._children:
        return False

    children = node._children[section.node_type]
    if len(children) == 0:
        return False

    padding = " " * indent
    output_stream.write(padding)
    output_stream.write(section.name)
    output_stream.write("\n")
    stub_generator = NODE_TYPE_TO_STUB_GENERATOR[section.node_type]
    for child in filter(lambda c: c.is_exported, children.values()):
        stub_generator(child, output_stream, indent)
    output_stream.write("\n")
    return True


def _generate_class_stub(class_node: ClassNode,
                         output_stream: StringIO, indent: int = 0):
    if len(class_node.bases) > 0:
        bases = "({})".format(', '.join(base.export_name for base in class_node.bases))
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
        output_stream.write("pass\n\n\n")


def _generate_constant_stub(constant_node: ConstantNode,
                            output_stream: StringIO, indent: int = 0):
    output_stream.write(
        "{indent}{name}: int\n".format(
            name=constant_node.export_name,
            indent=" " * indent
        )
    )


def _generate_enumeration_stub(enumeration_node: EnumerationNode,
                               output_stream: StringIO, indent: int = 0):
    for entry in enumeration_node.constants.values():
        _generate_constant_stub(entry, output_stream, indent)
    # Unnamed enumerations are skipped as definition
    if enumeration_node.export_name.endswith("<unnamed>"):
        output_stream.write("\n")
        return
    output_stream.write(
        "{indent}{name} = int  # One of [{entries}]\n\n".format(
            name=enumeration_node.export_name,
            entries=", ".join(entry.export_name
                              for entry in enumeration_node.constants.values()),
            indent=" " * indent
        )
    )


def _generate_function_stub(function_node: FunctionNode,
                            output_stream: StringIO, indent: int = 0):
    output_stream.write(
        "{indent}def {name}() -> None: ...\n".format(
            name=function_node.export_name,
            indent=" " * indent
        )
    )


def _generate_enums_from_classes_tree(class_node: ClassNode,
                                      output_stream: StringIO,
                                      indent: int = 0,
                                      class_name_prefix: str = ""):
    class_name_prefix = class_node.export_name + "_" + class_name_prefix
    for enum_node in class_node.enumerations.values():
        # Prefix enumeration and its entries with class name
        enum_node.export_name = class_name_prefix + enum_node.export_name
        for entry_node in enum_node.constants.values():
            entry_node.export_name = class_name_prefix + entry_node.export_name

        _generate_enumeration_stub(enum_node, output_stream, indent)
    for cls in class_node.classes.values():
        _generate_enums_from_classes_tree(cls, output_stream, indent,
                                          class_name_prefix)


StubGenerator = Callable[[ASTNode, StringIO, int], None]


NODE_TYPE_TO_STUB_GENERATOR: Dict[Type[ASTNode], StubGenerator] = {
    ClassNode: _generate_class_stub,
    ConstantNode: _generate_constant_stub,
    EnumerationNode: _generate_enumeration_stub,
    FunctionNode: _generate_function_stub
}
