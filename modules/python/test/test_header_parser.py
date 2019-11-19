#!/usr/bin/env python
import os
import sys
src_path = os.path.abspath(os.path.dirname(sys.argv[0]) + '/../src2')
sys.path.append(src_path)

from itertools import product

from hdr_parser import CppHeaderParser

from tests_common import NewOpenCVTests, unittest


format_to_modifiers = {
    '{type} {name}' : [],
    'const {type} {name}' : ['/C'],
    '{type} const {name}' : ['/C'],
    '{type}& {name}': ['/Ref'],
    '{type} & {name}': ['/Ref'],
    '{type} &{name}': ['/Ref'],
    'const {type}& {name}' : ['/C', '/Ref'],
    'const {type} & {name}' : ['/C', '/Ref'],
    'const {type} &{name}' : ['/C', '/Ref'],
    '{type} const& {name}' : ['/C', '/Ref'],
    '{type} const & {name}' : ['/C', '/Ref'],
    '{type} const &{name}' : ['/C', '/Ref'],
}


class CVHeaderParserTests(NewOpenCVTests):
    def test_find_next_token(self):
        test_string = 'begin_string;end_string'
        tokens = ('_', ';')

        parser = CppHeaderParser()

        token, pos = parser.find_next_token(test_string, (tokens[-1]))
        self.assertEqual(token, tokens[-1])
        self.assertEqual(pos, test_string.find(tokens[-1]))

        token, pos = parser.find_next_token(test_string, (tokens[0]))
        self.assertEqual(token, tokens[0])
        self.assertEqual(pos, test_string.find(tokens[0]))

        token, pos = parser.find_next_token(test_string, tokens)
        self.assertEqual(token, tokens[0])
        self.assertEqual(pos, test_string.find(tokens[0]))

        token, pos = parser.find_next_token(test_string, [")"])
        self.assertEqual(len(token), 0)
        self.assertEqual(pos, len(test_string))

        token, pos = parser.find_next_token(test_string, tokens,
                                            test_string.find(tokens[0]) + 1)
        self.assertEqual(token, tokens[-1])
        self.assertEqual(pos, test_string.find(tokens[-1]))

    def test_batch_replace(self):
        test_string = '([])'
        replace_map = ('(', '{'), (')', '}')

        test_string_clone = test_string

        parser = CppHeaderParser()
        s = parser.batch_replace(test_string, replace_map)
        self.assertEqual(s, '{[]}')
        self.assertEqual(test_string, test_string_clone)
        self.assertNotEqual(s, test_string)

    def test_get_macro_arg(self):
        parser = CppHeaderParser()

        test_line = 'TEST_MACRO(Arg1) rest of the string'
        argument, pos = parser.get_macro_arg(test_line, 0)
        self.assertEqual(argument, 'Arg1')
        self.assertEqual(pos, len(test_line.split(' ')[0]) - 1)

    def test_get_dotted_name(self):
        parser = CppHeaderParser()
        parser.wrap_mode = True

        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None]]
        self.assertEqual(parser.get_dotted_name('TestClass'), 'cv.TestClass')

        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None],
                              ['class', 'TestClass', True, True, None]]
        self.assertEqual(parser.get_dotted_name('func'), 'cv.TestClass.func')

        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None],
                              ['enum', 'EnumName', True, True, None]]
        self.assertEqual(parser.get_dotted_name('ENUM_VALUE'), 'cv.ENUM_VALUE')

        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None],
                              ['enum struct', 'EnumName', True, True, None]]
        self.assertEqual(parser.get_dotted_name('ENUM_VALUE'),
                         'cv.EnumName.ENUM_VALUE')

    def test_parse_enum_values(self):
        parser = CppHeaderParser()
        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None],
                              ['enum', 'TestEnum', True, True, None]]
        parser.wrap_mode = True

        self.assertListEqual(
            parser.parse_enum('FIRST, SECOND'),
            [['const cv.FIRST', '0', [], [], None, ''],
             ['const cv.SECOND', '1', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST = 3, SECOND'),
            [['const cv.FIRST', '3', [], [], None, ''],
             ['const cv.SECOND', '3+1', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST, SECOND, THIRD=34'),
            [['const cv.FIRST', '0', [], [], None, ''],
             ['const cv.SECOND', '1', [], [], None, ''],
             ['const cv.THIRD', '34', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST = 3, SECOND=2, THIRD=1'),
            [['const cv.FIRST', '3', [], [], None, ''],
             ['const cv.SECOND', '2', [], [], None, ''],
             ['const cv.THIRD', '1', [], [], None, '']]
        )

    def test_parse_enum_scoped_enum_values(self):
        parser = CppHeaderParser()
        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None],
                              ['enum struct', 'TestEnum', True, True, None]]
        parser.wrap_mode = True

        self.assertListEqual(
            parser.parse_enum('FIRST, SECOND'),
            [['const cv.TestEnum.FIRST', '0', [], [], None, ''],
             ['const cv.TestEnum.SECOND', '1', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST = 21, SECOND'),
            [['const cv.TestEnum.FIRST', '21', [], [], None, ''],
             ['const cv.TestEnum.SECOND', '21+1', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST, SECOND, THIRD=3'),
            [['const cv.TestEnum.FIRST', '0', [], [], None, ''],
             ['const cv.TestEnum.SECOND', '1', [], [], None, ''],
             ['const cv.TestEnum.THIRD', '3', [], [], None, '']]
        )

        self.assertListEqual(
            parser.parse_enum('FIRST = 3, SECOND=2, THIRD=1'),
            [['const cv.TestEnum.FIRST', '3', [], [], None, ''],
             ['const cv.TestEnum.SECOND', '2', [], [], None, ''],
             ['const cv.TestEnum.THIRD', '1', [], [], None, '']]
        )

    def test_parse_class_decl_general(self):
        parser = CppHeaderParser()

        for export_macro in ('CV_EXPORTS', 'CV_EXPORTS_W'):
            plain = 'class {0} TestClass'.format(export_macro)
            parsed = parser.parse_class_decl(plain)
            self.assertTupleEqual(parsed, ('TestClass', [], []))

            with_base = 'class {0} TestClass : public Base'.format(
                export_macro)
            parsed = parser.parse_class_decl(with_base)
            self.assertTupleEqual(parsed, ('TestClass', ['Base'], []))

            with_multiple_bases = 'class {0} TestClass : public Base1, ' \
                'Base2'.format(export_macro)
            parsed = parser.parse_class_decl(with_multiple_bases)
            self.assertTupleEqual(
                parsed, ('TestClass', ['Base1', 'Base2'], []))

    def test_parse_class_decl_map(self):
        parser = CppHeaderParser()

        plain = 'class CV_EXPORTS_W_MAP TestClass'
        parsed = parser.parse_class_decl(plain)
        self.assertTupleEqual(parsed, ('TestClass', [], ['/Map']))

        with_base = 'class CV_EXPORTS_W_MAP TestClass : public Base'
        parsed = parser.parse_class_decl(with_base)
        self.assertTupleEqual(parsed, ('TestClass', ['Base'], ['/Map']))

        with_multiple_bases = 'class CV_EXPORTS_W_MAP TestClass : public '\
            'Base1, Base2'
        parsed = parser.parse_class_decl(with_multiple_bases)
        self.assertTupleEqual(
            parsed, ('TestClass', ['Base1', 'Base2'], ['/Map']))

    def test_parse_class_decl_simple(self):
        parser = CppHeaderParser()

        plain = 'class CV_EXPORTS_W_SIMPLE TestClass'
        parsed = parser.parse_class_decl(plain)
        self.assertTupleEqual(parsed, ('TestClass', [], ['/Simple']))

        with_base = 'class CV_EXPORTS_W_SIMPLE TestClass : public Base'
        parsed = parser.parse_class_decl(with_base)
        self.assertTupleEqual(parsed, ('TestClass', ['Base'], ['/Simple']))

        with_multiple_bases = 'class CV_EXPORTS_W_SIMPLE TestClass : public '\
            'Base1, Base2'
        parsed = parser.parse_class_decl(with_multiple_bases)
        self.assertTupleEqual(
            parsed, ('TestClass', ['Base1', 'Base2'], ['/Simple']))

    def test_parse_class_decl_alias_export(self):
        parser = CppHeaderParser()

        plain = 'class CV_EXPORTS_AS(Plain) TestClass'
        parsed = parser.parse_class_decl(plain)
        self.assertTupleEqual(parsed, ('TestClass', [], ['=Plain']))

        with_base = 'class CV_EXPORTS_AS(WithBase) TestClass : public Base'
        parsed = parser.parse_class_decl(with_base)
        self.assertTupleEqual(parsed, ('TestClass', ['Base'], ['=WithBase']))

        with_multiple_bases = 'class CV_EXPORTS_AS(MultiBase) TestClass : ' \
            'public Base1, Base2'
        parsed = parser.parse_class_decl(with_multiple_bases)
        self.assertTupleEqual(
            parsed, ('TestClass', ['Base1', 'Base2'], ['=MultiBase']))

    def test_parse_stmt_namespace(self):
        parser = CppHeaderParser()

        test_ns = 'namespace test'
        parser.block_stack = [['file', 'file_name', True, True, None]]
        parser.wrap_mode = True
        parsed = parser.parse_stmt(test_ns, '{')
        self.assertTupleEqual(parsed, ('namespace', 'test', True, None))

        unnamed_ns = 'namespace'
        parsed = parser.parse_stmt(unnamed_ns, '{')
        self.assertTupleEqual(parsed, ('namespace', '<unnamed>', True, None))

    def test_parse_stmt_enum_declaration(self):
        parser = CppHeaderParser()
        parser.wrap_mode = True
        parser.block_stack = [['file', 'file_name', True, True, None],
                              ['namespace', 'cv', True, True, None]]

        old_style_enum = 'enum TestEnum'
        parsed = parser.parse_stmt(old_style_enum, '{')
        self.assertTupleEqual(parsed, ('enum', 'TestEnum', True, None))

        enum_struct = 'enum struct TestEnum'
        parsed = parser.parse_stmt(enum_struct, '{')
        self.assertTupleEqual(parsed, ('enum struct', 'TestEnum', True, None))

        enum_class = 'enum class TestEnum'
        parsed = parser.parse_stmt(enum_class, '{')
        self.assertTupleEqual(parsed, ('enum class', 'TestEnum', True, None))

        # scoped_enum_with_type = 'enum class TestEnum : int'
        # parsed = parser.parse_stmt(scoped_enum_with_type, '{'),
        # self.assertTupleEqual(parsed, ('enum class', 'TestEnum', True, None))

    def test_parse_arg_nameless_args(self):
        parser = CppHeaderParser()

        types = {
            'double': 'double',
            'Scalar': 'Scalar',
            'InputArray': 'InputArray',
            'OutputArray': 'OutputArray',
            'UMat': 'UMat',
            'cv::Mat': 'Mat',
            'std::vector<double>': 'vector_double',
            'std::pair<int, double>': 'pair_int_and_double',
            '::test': '_test',
            '::test<std::Pair<int, double>>': '_test_Pair_int_and_double'
        }
        indices = (0, 3, 2)
        for atype, fmt_str, index in product(types, format_to_modifiers, indices):
            str_to_parse = fmt_str.format(type=atype, name='')
            self.assertTupleEqual(
                parser.parse_arg(str_to_parse, index),
                (types[atype], 'arg{0}'.format(index),
                 format_to_modifiers[fmt_str], index + 1),
                msg="Can't parse {0}".format(str_to_parse)
            )

    @unittest.expectedFailure
    def test_parse_arg_custom_namespace(self):
        parser = CppHeaderParser()

        types = {
            'detail::Type': 'Type',
            '::test<detail::Pair<int, double>>': '_test_Pair_int_and_double'
        }
        format_to_modifiers = {
            '{type}': [],
            'const {type}': ['/C'],
            '{type} const': ['/C'],
            '{type}&': ['/Ref'],
            '{type} &': ['/Ref'],
            'const {type}&': ['/C', '/Ref'],
            'const {type} &': ['/C', '/Ref'],
            '{type} const&': ['/C', '/Ref'],
            '{type} const &': ['/C', '/Ref']
        }
        indices = (0, 3, 2)
        for atype, fmt_str, index in product(types, format_to_modifiers, indices):
            str_to_parse = fmt_str.format(type=atype, name='')
            self.assertTupleEqual(
                parser.parse_arg(str_to_parse, index),
                (types[atype], 'arg{0}'.format(index),
                 format_to_modifiers[fmt_str], index + 1),
                msg="Can't parse {0}".format(str_to_parse)
            )

    def test_parse_arg_named_args(self):
        parser = CppHeaderParser()

        types = {
            'double': 'double',
            'Scalar': 'Scalar',
            'InputArray': 'InputArray',
            'OutputArray': 'OutputArray',
            'UMat': 'UMat',
            'cv::Mat': 'Mat',
            'std::vector<double>': 'vector_double',
            'std::pair<int, double>': 'pair_int_and_double',
            '::test': '_test',
            '::test<std::Pair<int, double>>': '_test_Pair_int_and_double'
        }
        indices = (0, 3, 2)
        for atype, fmt_str, index in product(types, format_to_modifiers, indices):
            str_to_parse = fmt_str.format(type=atype, name='argument')
            self.assertTupleEqual(
                parser.parse_arg(str_to_parse, index),
                (types[atype], 'argument',
                 format_to_modifiers[fmt_str], index),
                msg="Can't parse {0}".format(str_to_parse)
            )

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
