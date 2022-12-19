from typing import Tuple, Dict, Sequence

from .aliases import ALIASES


def replace_template_parameters_with_placeholders(string: str) \
        -> Tuple[str, Tuple[str, ...]]:
    """Replaces template parameters with `format` placeholders for all template
    instantiations in provided string.
    Only outermost template parameters are replaced.

    Args:
        string (str): input string containing C++ template instantiations

    Returns:
        tuple[str, tuple[str, ...]]: string with '{}' placeholders  template
            instead of instantiation types and a tuple of extracted types.

    >>> template_string, args = replace_template_parameters_with_placeholders(
    ...     "std::vector<cv::Point<int>>, test<int>"
    ... )
    >>> template_string.format(*args) == "std::vector<cv::Point<int>>, test<int>"
    True

    >>> replace_template_parameters_with_placeholders(
    ...     "cv::util::variant<cv::GRunArgs, cv::GOptRunArgs>"
    ... )
    ('cv::util::variant<{}>', ('cv::GRunArgs, cv::GOptRunArgs',))

    >>> replace_template_parameters_with_placeholders("vector<Point<int>>")
    ('vector<{}>', ('Point<int>',))

    >>> replace_template_parameters_with_placeholders(
    ...     "vector<Point<int>>, vector<float>"
    ... )
    ('vector<{}>, vector<{}>', ('Point<int>', 'float'))

    >>> replace_template_parameters_with_placeholders("string without templates")
    ('string without templates', ())
    """

    template_brackets_indices = []
    template_instantiations_count = 0
    template_start_index = 0
    for i, c in enumerate(string):
        if c == "<":
            template_instantiations_count += 1
            if template_instantiations_count == 1:
                # + 1 - because left bound is included in substring range
                template_start_index = i + 1
        elif c == ">":
            template_instantiations_count -= 1
            assert template_instantiations_count >= 0, \
                "Provided string is ill-formed. There are more '>' than '<'."
            if template_instantiations_count == 0:
                template_brackets_indices.append((template_start_index, i))
    assert template_instantiations_count == 0, \
        "Provided string is ill-formed. There are more '<' than '>'."
    template_args = []  # type: list[str]
    # Reversed loop is required to preserve template start/end indices
    for i, j in reversed(template_brackets_indices):
        template_args.insert(0, string[i:j])
        string = string[:i] + "{}" + string[j:]
    return string, tuple(template_args)


def get_template_instantiation_type(typename: str) -> str:
    """Extracts outermost template instantiation type from provided string

    Args:
        typename (str): String containing C++ template instantiation.

    Returns:
        str: String containing template instantiation type

    >>> get_template_instantiation_type("std::vector<cv::Point<int>>")
    'cv::Point<int>'
    >>> get_template_instantiation_type("std::vector<uchar>")
    'uchar'
    >>> get_template_instantiation_type("std::map<int, float>")
    'int, float'
    >>> get_template_instantiation_type("uchar")
    Traceback (most recent call last):
    ...
    ValueError: typename ('uchar') doesn't contain template instantiations
    >>> get_template_instantiation_type("std::vector<int>, std::vector<float>")
    Traceback (most recent call last):
    ...
    ValueError: typename ('std::vector<int>, std::vector<float>') contains more than 1 template instantiation
    """

    _, args = replace_template_parameters_with_placeholders(typename)
    if len(args) == 0:
        raise ValueError(
            "typename ('{}') doesn't contain template instantiations".format(typename)
        )
    if len(args) > 1:
        raise ValueError(
            "typename ('{}') contains more than 1 template instantiation".format(typename)
        )
    return args[0]


def normalize_ctype_name(typename: str) -> str:
    """Normalizes C++ name by removing unnecessary namespace prefixes and possible
    pointer/reference qualification

    Args:
        typename (str): Name of the C++ type for normalization

    Returns:
        str: Normalized C++ type name.

    >>> normalize_ctype_name("std::vector<cv::Point2f>&")
    'vector<cv::Point2f>'
    >>> normalize_ctype_name("Ptr<AKAZE>")
    'AKAZE'
    >>> normalize_ctype_name("AKAZE::DescriptorType")
    'AKAZE::DescriptorType'
    >>> normalize_ctype_name("std::vector<Mat>")
    'vector<Mat>'
    >>> normalize_ctype_name("std::string")
    'string'
    """
    for prefix_to_remove in ("cv", "std"):
        if typename.startswith(prefix_to_remove):
            typename = typename[len(prefix_to_remove):]
    typename = typename.replace("::", "_").lstrip("_")
    if typename.endswith('&'):
        typename = typename[:-1]
    return typename.strip()


_CTYPE_TO_PYTYPE_DIRECT_SUBSTITUTION_MAP = {
    "char": "str",
    "uchar": "int",
    "unsigned": "int",
    "String": "str",
    "string": "str",
    "c_string": "str",
    "double": "float",
    "int64": "int",
    "size_t": "int",
    "void": "None",
    "vector<uchar>": "numpy.ndarray[Any, numpy.dtype[numpy.uint8]]",
    "vector_uchar": "numpy.ndarray[Any, numpy.dtype[numpy.uint8]]",
    "GMat2": "tuple[GMat, GMat]",
    "GOpaque": "GOpaqueT",
    "GArray": "GArrayT",
    "GCompileArgs": "Sequence[GCompileArg]",
    "GTypesInfo": "Sequence[GTypeInfo]",
    "GRunArgs": "Sequence[GRunArg]",
    "GMetaArgs": "Sequence[GMetaArg]",
    "GProtoArgs": "Sequence[GProtoArg]",
    "GOptRunArgs": "Sequence[GOptRunArg]",
    "detail_ExtractArgsCallback": "Callable[[Sequence[GTypeInfo]], Sequence[GRunArg]]",
    "detail_ExtractMetaCallback": "Callable[[Sequence[GTypeInfo]], Sequence[GMetaArg]]",
    "Prims": "Sequence[Prim]",
    "LayerId": "DictValue",
    "flann_IndexParams": "dict[str, bool | int | float | str]",
    "flann_SearchParams": "dict[str, bool | int | float | str]",
    "cvflann_flann_distance_t": "int",
    "cvflann_flann_algorithm_t": "int",
}


def is_tuple_type(typename: str) -> bool:
    return typename.startswith("tuple") or typename.startswith("pair")


def is_sequence_type(typename: str) -> bool:
    return typename.startswith("vector")


def is_pointer_type(typename: str) -> bool:
    return typename.endswith("Ptr") or typename.endswith("*") \
        or typename.startswith("Ptr")


def _is_template_instantiation(typename: str) -> bool:
    """Fast but unreliable check whenever provided typename is a template
    instantiation

    Args:
        typename (str): typename to check against template instantiation.

    Returns:
        bool: True if provided `typename` contains template instantiation,
            False otherwise
    """

    if "<" in typename:
        assert ">" in typename, \
            "Wrong template class instantiation: {}. '>' is missing".format(typename)
        return True
    return False


def convert_template_arguments_to_pytypes_arguments(template_args_str: str,
                                                    known_classes: Dict[str, str],
                                                    known_enumerations: Dict[str, str]) \
        -> Sequence[str]:
    pytypes = []
    # If template arguments contains types that are also templates
    # - replace it with format placeholder and than reconstruct original type.
    # It covers cases when inner template types have several template params.
    # e.g. std::tuple<std::variant<int, Point<int>, int, std::vector<int>>
    template_args_str, templated_args_types = replace_template_parameters_with_placeholders(
        template_args_str
    )
    template_index = 0
    for template_arg in template_args_str.split(","):
        template_arg = template_arg.strip()
        # Check if this arg requires type substitution
        if _is_template_instantiation(template_arg):
            template_arg = template_arg.format(templated_args_types[template_index])
            template_index += 1
        pytypes.append(
            convert_ctype_name_to_pytype_name(template_arg, known_classes,
                                              known_enumerations)
        )
    return pytypes


def convert_ctype_name_to_pytype_name(typename: str,
                                      known_classes: Dict[str, str],
                                      known_enumerations: Dict[str, str]) -> str:
    """Converts C++ type name to corresponding Python type name

    Args:
        typename (str): C++ type name to convert.
        known_classes (Dict[str]): Mapping between C++ classes names and their
            names exposed to Python.
        known_enumerations (Dict[str]): Mapping between C++ enumerations names
            and their names exposed to Python.

    Returns:
        str: typename that should be exposed to Python
    """

    original_ctype_name = typename
    typename = normalize_ctype_name(typename.strip())

    # if typename is one of the built-in Python types
    if typename in ("float", "int", "bool"):
        return typename

    # if typename has a direct substitution
    pytype = _CTYPE_TO_PYTYPE_DIRECT_SUBSTITUTION_MAP.get(typename)
    if pytype is not None:
        return pytype

    # if typename is a known alias
    type_alias = ALIASES.get(typename)
    if type_alias is not None:
        return type_alias.export_name

    # explicit handling of special G-Api Types
        # GAPI types
    if typename.startswith("GArray_") or typename.startswith("GArray<"):
        return "GArrayT"
    if typename.startswith("GOpaque_") or typename.startswith("GOpaque<"):
        return "GOpaqueT"
    if typename.startswith("util_variant"):
        variant_types = get_template_instantiation_type(typename)
        return " | ".join(
            convert_template_arguments_to_pytypes_arguments(variant_types,
                                                            known_classes,
                                                            known_enumerations)
        )

    if is_pointer_type(typename):
        # Case for "type*", "type_Ptr", "typePtr"
        for suffix in ("*", "_Ptr", "Ptr"):
            if typename.endswith(suffix):
                return convert_ctype_name_to_pytype_name(typename[:-len(suffix)],
                                                         known_classes,
                                                         known_enumerations)
        # Case Ptr<Type>
        if _is_template_instantiation(typename):
            return convert_ctype_name_to_pytype_name(
                get_template_instantiation_type(typename), known_classes,
                known_enumerations
            )
        # Case Ptr_Type
        return convert_ctype_name_to_pytype_name(
            typename.split("_", maxsplit=1)[-1], known_classes, known_enumerations
        )

    # if typename refers to a sequence type
    if is_sequence_type(typename):
        # Recursively convert sequence element type
        if _is_template_instantiation(typename):
            inner_sequence_type = convert_ctype_name_to_pytype_name(
                get_template_instantiation_type(typename), known_classes,
                known_enumerations
            )
        else:
            # Handle vector_Type cases
            # maxsplit=1 is required to handle sequence of sequence e.g:
            # vector_vector_Mat -> Sequence[Sequence[Mat]]
            inner_sequence_type = convert_ctype_name_to_pytype_name(
                typename.split("_", 1)[-1], known_classes, known_enumerations
            )
        return "Sequence[" + inner_sequence_type + "]"

    if is_tuple_type(typename):
        tuple_types = get_template_instantiation_type(typename)
        return "tuple[{}]".format(", ".join(
            convert_template_arguments_to_pytypes_arguments(
                tuple_types, known_classes, known_enumerations
            )
        ))

    return typename


if __name__ == "__main__":
    import doctest
    doctest.testmod()
