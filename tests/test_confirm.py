from unittest.mock import patch
from agentforge.confirm import InteractiveConfirm


def test_confirm_yes(tmp_path):
    """用户输入 Y，文件被写入。"""
    files = {"test.yaml": "content: hello"}
    with patch("builtins.input", return_value="Y"):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "test.yaml").read_text() == "content: hello"
    assert result == {"test.yaml": "accepted"}


def test_confirm_empty_input_means_yes(tmp_path):
    """空回车 = Y。"""
    files = {"a.py": "# code"}
    with patch("builtins.input", return_value=""):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "a.py").read_text() == "# code"
    assert result == {"a.py": "accepted"}


def test_confirm_no_skips(tmp_path):
    """用户输入 n，文件不写入。"""
    files = {"skip.yaml": "content: skip"}
    with patch("builtins.input", return_value="n"):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert not (tmp_path / "skip.yaml").exists()
    assert result == {"skip.yaml": "rejected"}


def test_confirm_multiple_files(tmp_path):
    """多个文件逐个确认。"""
    files = {
        "challenge.yaml": "challenge: test",
        "benchmark.py": "# bench",
        "test_suite.py": "# tests",
    }
    responses = iter(["Y", "n", "Y"])
    with patch("builtins.input", side_effect=responses):
        confirm = InteractiveConfirm(workdir=tmp_path)
        result = confirm.confirm_each(files)
    assert (tmp_path / "challenge.yaml").exists()
    assert not (tmp_path / "benchmark.py").exists()
    assert (tmp_path / "test_suite.py").exists()
    assert result == {
        "challenge.yaml": "accepted",
        "benchmark.py": "rejected",
        "test_suite.py": "accepted",
    }
