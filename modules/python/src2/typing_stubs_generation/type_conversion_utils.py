from typing import Tuple, Dict


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
    typename = typename.replace("::", "_")
    if typename.endswith('&'):
        typename = typename[:-1]
    return typename.strip()


# def convert_ctype_name_to_pytype_name(typename: str,
#                                       known_classes: Dict[str, str],
#                                       known_enumerations: Dict[str, str]) -> str:
#     """Converts C++ type name to corresponding Python type name

#     Args:
#         typename (str): C++ type name to convert.
#         known_classes (Dict[str]): Mapping between C++ classes names and their
#             names exposed to Python.
#         known_enumerations (Dict[str]): Mapping between C++ enumerations names
#             and their names exposed to Python.

#     Returns:
#         str: typename that should be exposed to Python

#     >>> convert_ctype_name_to_pytype_name()
#     """

#     original_ctype_name = typename
#     typename = normalize_ctype_name(typename.strip())


if __name__ == "__main__":
    import doctest
    doctest.testmod()
