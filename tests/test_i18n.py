import ast
from pathlib import Path
import re
import unittest

from streamlit.testing.v1 import AppTest

from animalact2.dashboard.ui.i18n import ZH


class LanguageTests(unittest.TestCase):
    def test_all_pages_in_both_languages(self):
        for language in ("en", "zh"):
            for page in ("video", "results", "evaluation"):
                with self.subTest(language=language, page=page):
                    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=120)
                    app.query_params.update(page=page, lang=language)
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error, [item.value for item in app.error])
                    self.assertEqual(app.session_state["nav"], page.title())
                    self.assertEqual(app.session_state["ui_language"], language)
                    if language == "zh":
                        self.assertRegex(app.title[0].value, "[\u4e00-\u9fff]")
        app.session_state["ui_language"] = "en"
        app.run()
        self.assertEqual(app.session_state["nav"], "Evaluation")

    def test_legacy_entry_and_unified_model_selection(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=120)
        app.query_params.update(page="proposal", lang="zh")
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state["nav"], "Evaluation")
        self.assertEqual(app.radio(key="nav").options, ["视频识别", "识别结果", "实验评估"])
        app.radio(key="nav").set_value("Video").run()
        from animalact2.dashboard.ui.services.resources import available_models
        for model in available_models():
            app.selectbox(key="model_key").set_value(model).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(app.text_input)

    def test_display_strings_have_chinese_entries(self):
        root = Path("animalact2/dashboard/ui")
        for path in [root / "app.py", *(root / "pages").glob("*.py"), root / "components/ui.py"]:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "t" or not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                text = node.args[0].value
                if isinstance(text, str) and re.search("[A-Za-z]", text) and "<div" not in text:
                    self.assertIn(text, ZH, f"{path}:{node.lineno}")


if __name__ == "__main__":
    unittest.main()
