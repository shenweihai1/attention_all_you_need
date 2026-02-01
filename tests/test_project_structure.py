"""
Tests for verifying the project structure is correct.
"""

import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestProjectStructure:
    """Tests for verifying the project directory structure."""

    def test_src_directory_exists(self):
        """Test that src/ directory exists."""
        src_path = os.path.join(PROJECT_ROOT, "src")
        assert os.path.isdir(src_path), "src/ directory should exist"

    def test_tests_directory_exists(self):
        """Test that tests/ directory exists."""
        tests_path = os.path.join(PROJECT_ROOT, "tests")
        assert os.path.isdir(tests_path), "tests/ directory should exist"

    def test_docs_directory_exists(self):
        """Test that docs/ directory exists."""
        docs_path = os.path.join(PROJECT_ROOT, "docs")
        assert os.path.isdir(docs_path), "docs/ directory should exist"

    def test_configs_directory_exists(self):
        """Test that configs/ directory exists."""
        configs_path = os.path.join(PROJECT_ROOT, "configs")
        assert os.path.isdir(configs_path), "configs/ directory should exist"

    def test_src_init_exists(self):
        """Test that src/__init__.py exists."""
        init_path = os.path.join(PROJECT_ROOT, "src", "__init__.py")
        assert os.path.isfile(init_path), "src/__init__.py should exist"

    def test_tests_init_exists(self):
        """Test that tests/__init__.py exists."""
        init_path = os.path.join(PROJECT_ROOT, "tests", "__init__.py")
        assert os.path.isfile(init_path), "tests/__init__.py should exist"

    def test_configs_init_exists(self):
        """Test that configs/__init__.py exists."""
        init_path = os.path.join(PROJECT_ROOT, "configs", "__init__.py")
        assert os.path.isfile(init_path), "configs/__init__.py should exist"

    def test_requirements_exists(self):
        """Test that requirements.txt exists."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        assert os.path.isfile(req_path), "requirements.txt should exist"

    def test_requirements_not_empty(self):
        """Test that requirements.txt is not empty."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r") as f:
            content = f.read()
        assert len(content.strip()) > 0, "requirements.txt should not be empty"

    def test_requirements_has_torch(self):
        """Test that requirements.txt includes torch."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r") as f:
            content = f.read()
        assert "torch" in content.lower(), "requirements.txt should include torch"

    def test_requirements_has_pytest(self):
        """Test that requirements.txt includes pytest."""
        req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        with open(req_path, "r") as f:
            content = f.read()
        assert "pytest" in content.lower(), "requirements.txt should include pytest"

    def test_base_config_json_exists(self):
        """Test that base_config.json exists."""
        config_path = os.path.join(PROJECT_ROOT, "configs", "base_config.json")
        assert os.path.isfile(config_path), "configs/base_config.json should exist"

    def test_transformer_config_module_exists(self):
        """Test that transformer_config.py exists."""
        config_path = os.path.join(PROJECT_ROOT, "configs", "transformer_config.py")
        assert os.path.isfile(config_path), "configs/transformer_config.py should exist"
