from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.data import prepare_data


class TestPrepareData:
    def test_no_config(self, tmp_path: Path):
        """data_config=None 时什么都不做。"""
        prepare_data(tmp_path, None)

    def test_data_exists_skips_download(self, tmp_path: Path):
        """数据已存在时跳过下载。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.bin").write_bytes(b"fake")
        prepare_data(tmp_path, {"path": "data"})

    def test_missing_env_var(self, tmp_path: Path):
        """缺少必要环境变量时报错。"""
        with pytest.raises(RuntimeError, match="缺少环境变量"):
            prepare_data(tmp_path, {
                "path": "data",
                "requires_env": ["NONEXISTENT_VAR_12345"],
            })

    def test_auto_download_with_script(self, tmp_path: Path):
        """auto 模式下找到 prepare.py 并执行。"""
        # data 脚本放在 data/ 下，但实际数据目录是 mydata/（空的）
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        script = data_dir / "prepare.py"
        out_dir = tmp_path / "mydata"
        script.write_text(
            f"from pathlib import Path\n"
            f"Path('{out_dir}').mkdir(exist_ok=True)\n"
            f"(Path('{out_dir}') / 'ready.txt').write_text('ok')\n"
        )
        prepare_data(tmp_path, {"path": "mydata", "source": "auto"})
        assert (out_dir / "ready.txt").exists()
