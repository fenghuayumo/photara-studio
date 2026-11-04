import unittest

from format_code import validate_hlsl


class ShaderSafetyTests(unittest.TestCase):
    def test_spacing_only(self):
        validate_hlsl('if(a+b) return;\n', 'if (a + b)\n    return;\n')

    def test_reject_changes(self):
        cases = [
            ('f("a,b");', 'f("a, b");'),
            ('a + +b;', 'a++b;'),
            ('#define F(x) x\n', '#define F (x) x\n'),
            ('#include "a"\n#include "b"\n', '#include "b"\n#include "a"\n'),
            ('f(__LINE__);\ng();\n', '\nf(__LINE__);g();\n'),
            ('#define F x\\\ny\n', '#define F x y\n'),
        ]
        for before, after in cases:
            with self.subTest(before=before), self.assertRaises(ValueError):
                validate_hlsl(before, after)


if __name__ == '__main__':
    unittest.main()
