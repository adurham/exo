import os
import tempfile
from pathlib import Path

import pytest

from exo.utils.fs import (
    delete_if_exists,
    ensure_directory_exists,
    ensure_parent_directory_exists,
    make_temp_path,
)


class TestDeleteIfExists:
    def test_delete_if_exists_file_exists(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            assert os.path.exists(tmp_path)
            delete_if_exists(tmp_path)
            assert not os.path.exists(tmp_path)

    def test_delete_if_exists_file_not_exists(self):
        # Should not raise an error
        delete_if_exists("/nonexistent/file/path.txt")

    def test_delete_if_exists_with_path_object(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
            assert tmp_path.exists()
            delete_if_exists(tmp_path)
            assert not tmp_path.exists()


class TestEnsureParentDirectoryExists:
    def test_ensure_parent_directory_exists_creates_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "file.txt"
            assert not file_path.parent.exists()
            ensure_parent_directory_exists(str(file_path))
            assert file_path.parent.exists()

    def test_ensure_parent_directory_exists_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            file_path = subdir / "file.txt"
            ensure_parent_directory_exists(str(file_path))
            assert subdir.exists()

    def test_ensure_parent_directory_exists_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "level1" / "level2" / "level3" / "file.txt"
            ensure_parent_directory_exists(str(file_path))
            assert file_path.parent.exists()
            assert (file_path.parent.parent).exists()

    def test_ensure_parent_directory_exists_with_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "file.txt"
            ensure_parent_directory_exists(file_path)
            assert file_path.parent.exists()


class TestEnsureDirectoryExists:
    def test_ensure_directory_exists_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "newdir"
            assert not new_dir.exists()
            ensure_directory_exists(str(new_dir))
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_directory_exists_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_dir = Path(tmpdir) / "existing"
            existing_dir.mkdir()
            ensure_directory_exists(str(existing_dir))
            assert existing_dir.exists()

    def test_ensure_directory_exists_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "level1" / "level2" / "level3"
            ensure_directory_exists(str(nested_dir))
            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_ensure_directory_exists_with_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "newdir"
            ensure_directory_exists(new_dir)
            assert new_dir.exists()


class TestMakeTempPath:
    def test_make_temp_path_creates_temp_directory(self):
        path = make_temp_path("test.txt")
        assert isinstance(path, str)
        assert "test.txt" in path
        # The directory should exist
        assert os.path.exists(os.path.dirname(path))

    def test_make_temp_path_creates_unique_paths(self):
        path1 = make_temp_path("test1.txt")
        path2 = make_temp_path("test2.txt")
        assert path1 != path2
        assert os.path.dirname(path1) != os.path.dirname(path2)

