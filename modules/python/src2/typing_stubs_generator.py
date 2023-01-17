"""Contains a class used to resolve compatibility issues with old Python versions.

Typing stubs generation is available starting from Python 3.6 only.
For other versions all calls to functions are noop.
"""

import sys
import warnings


if sys.version_info >= (3, 6):
    from typing import Dict, Set, Any, Sequence

    from pathlib import Path

    from typing_stubs_generation import (
        generate_typing_stubs,
        NamespaceNode,
        EnumerationNode,
        SymbolName,
        ClassNode,
        create_function_node,
        create_class_node,
        find_class_node,
        resolve_enum_scopes
    )

    import functools

    class FailuresWrapper:
        def __init__(self, *, exceptions_as_warnings = True) -> None:
            self.has_failure = False
            self.exceptions_as_warnings = exceptions_as_warnings

        def wrap_exceptions_as_warnings(self, original_func=None, *,
                                        ret_type_on_failure=None):
            def parametrized_wrapper(func):
                @functools.wraps(func)
                def wrapped_func(*args, **kwargs):
                    if self.has_failure:
                        if ret_type_on_failure is None:
                            return None
                        return ret_type_on_failure()

                    try:
                        ret_type = func(*args, **kwargs)
                    except Exception as e:
                        self.has_failure = True
                        warnings.warn(
                            'Typings stubs generation has failed. Reason: {}'.format(e)
                        )
                        if ret_type_on_failure is None:
                            return None
                        return ret_type_on_failure()
                    return ret_type

                if self.exceptions_as_warnings:
                    return wrapped_func
                else:
                    return original_func

            if original_func:
                return parametrized_wrapper(original_func)
            return parametrized_wrapper

    failures_wrapper = FailuresWrapper(exceptions_as_warnings=True)

    class ClassNodeStub:
        def add_base(self, base_node):
            pass

    class TypingStubsGenerator:
        def __init__(self):
            self.cv_root = NamespaceNode("cv", export_name="cv2")
            self.exported_enums = {}  # type: Dict[SymbolName, EnumerationNode]
            self.type_hints_ignored_functions = set()  # type: Set[str]

        @failures_wrapper.wrap_exceptions_as_warnings
        def add_enum(self, symbol_name, is_scoped_enum, entries):
            # type: (SymbolName, bool, Dict[str, str]) -> None
            if symbol_name in self.exported_enums:
                assert symbol_name.name == "<unnamed>", \
                    "Trying to export 2 enums with same symbol " \
                    "name: {}".format(symbol_name)
                enumeration_node = self.exported_enums[symbol_name]
            else:
                enumeration_node = EnumerationNode(symbol_name.name,
                                                   is_scoped_enum)
                self.exported_enums[symbol_name] = enumeration_node
            for entry_name, entry_value in entries.items():
                enumeration_node.add_constant(entry_name, entry_value)

        @failures_wrapper.wrap_exceptions_as_warnings
        def add_ignored_function_name(self, function_name):
            # type: (str) -> None
            self.type_hints_ignored_functions.add(function_name)

        @failures_wrapper.wrap_exceptions_as_warnings
        def create_function_node(self, func_info):
            # type: (Any) -> None
            create_function_node(self.cv_root, func_info)

        @failures_wrapper.wrap_exceptions_as_warnings(ret_type_on_failure=ClassNodeStub)
        def find_class_node(self, class_info, namespaces):
            # type: (Any, Sequence[str]) -> ClassNode
            return find_class_node(self.cv_root, class_info.full_original_name, namespaces)

        @failures_wrapper.wrap_exceptions_as_warnings(ret_type_on_failure=ClassNodeStub)
        def create_class_node(self, class_info, namespaces):
            # type: (Any, Sequence[str]) -> ClassNode
            return create_class_node(self.cv_root, class_info, namespaces)

        @failures_wrapper.wrap_exceptions_as_warnings
        def generate(self, output_path):
            # type: (str) -> None
            resolve_enum_scopes(self.cv_root, self.exported_enums)
            generate_typing_stubs(self.cv_root, Path(output_path))

else:
    class ClassNode:
        def add_base(self, base_node):
            pass

    class TypingStubsGenerator:
        def __init__(self):
            self.type_hints_ignored_functions = set()  # type: Set[str]

        def add_enum(self, symbol_name, is_scoped_enum, entries):
            pass

        def add_ignored_function_name(self, function_name):
            pass

        def create_function_node(self, func_info):
            pass

        def create_class_node(self, class_info, namespaces):
            return ClassNode()

        def find_class_node(self, class_info, namespaces):
            return ClassNode()

        def generate(self, output_path):
            pass
