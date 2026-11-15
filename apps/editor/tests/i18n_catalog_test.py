"""Check the Studio catalog and literal translation calls without dependencies."""
import json
from pathlib import Path
import re
import unittest

EDITOR = Path(__file__).resolve().parents[1]
LITERAL = r'"(?:[^"\\]|\\.)*"'
SEQUENCE = LITERAL + r'(?:\s*' + LITERAL + r')*'
ENTRY = re.compile(r'\{\s*(' + SEQUENCE + r')\s*,\s*(' + SEQUENCE
                   + r')\s*,\s*(' + SEQUENCE + r')\s*,\s*(' + SEQUENCE + r')\s*\}')
# These names, abbreviations and numeric diagnostics intentionally stay unchanged.
NEUTRAL = {
    'JPEG %d', 'NMS %.2f', 'DCC', 'glTF', 'Photara ASFM', 'COLMAP',
    'Nerfstudio / Blender', 'OpenMVS', 'Niantic SPZ.',
    'Khronos KHR_gaussian_splatting GLB.', 'VRAM %s / %s  %.0f%%',
    'RealityCapture', 'ADC IGS', 'ADC Plus', 'TSDF', 'PAM', 'sRGB',
    'RGB', '3DGS', 'Photara Studio', 'PNG', 'Delaunay',
}
FORMAT = re.compile(r'(?<![0-9])%(?:[-+#0 ]*\d*(?:\.\d+)?(?:hh|ll|[hljztL])?[diuoxXfFeEgGaAcsp]|%)')


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = [tuple(''.join(json.loads(t) for t in re.findall(LITERAL, x)) for x in m.groups())
                       for m in ENTRY.finditer((EDITOR / 'i18n.cpp').read_text(encoding="utf-8"))]
        cls.keys = {e[0] for e in cls.entries}

    def test_unique_complete_translations(self):
        self.assertGreater(len(self.entries), 1500)
        self.assertEqual(len(self.entries), len(self.keys), 'Duplicate translation key')
        for entry in self.entries:
            for language, translation in zip(('zh-CN', 'ja', 'ko'), entry[1:]):
                with self.subTest(key=entry[0], language=language):
                    self.assertTrue(translation.strip())
                    self.assertNotIn('\ufffd', translation)
                    if entry[0] not in NEUTRAL:
                        self.assertNotEqual(entry[0], translation)
                    self.assertEqual(FORMAT.findall(entry[0]), FORMAT.findall(translation))

    def test_shortcut_labels_are_registered(self):
        row = re.compile(r'\{\s*' + LITERAL + r',\s*(' + LITERAL
                         + r'),\s*(' + LITERAL + r'),\s*\{ImGuiKey_')
        for match in row.finditer((EDITOR / 'shortcuts.hpp').read_text(encoding="utf-8")):
            for literal in match.groups():
                self.assertTrue(json.loads(literal) in self.keys, f"Missing shortcut label: {literal}")

    def test_adjacent_cpp_literals(self):
        entry = '{"a" "b", "中", "日", "韓"}'
        match = ENTRY.fullmatch(entry)
        self.assertIsNotNone(match)
        self.assertEqual(''.join(json.loads(t) for t in re.findall(LITERAL, match[1])), 'ab')

    def test_literal_translation_calls_are_registered(self):
        sequence = LITERAL + r'(?:\s*' + LITERAL + r')*'
        call = re.compile(r'(?:\btr|i18n::id|theme::caption|preference_heading)\(\s*('
                          + sequence + ')')
        for source in EDITOR.rglob('*.cpp'):
            if source.name == 'i18n.cpp' or any(p in source.parts for p in ('tests', 'splat_render')):
                continue
            for match in call.finditer(source.read_text(encoding="utf-8")):
                try:
                    key = ''.join(json.loads(x) for x in re.findall(LITERAL, match[1]))
                except ValueError:
                    # Character escape labels such as "\\xef..." are UI icons.
                    continue
                with self.subTest(file=source.name, key=key):
                    self.assertTrue(key in self.keys, f"Missing translation key: {key}")


if __name__ == '__main__':
    unittest.main()
