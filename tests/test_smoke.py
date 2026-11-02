import signal_factory


def test_package_imports() -> None:
    assert signal_factory.__version__ == "0.1.0"
