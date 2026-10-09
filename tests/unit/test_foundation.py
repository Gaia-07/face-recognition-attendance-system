"""Foundational smoke tests verifying environment and package installation."""

import fras


def test_package_version() -> None:
    assert fras.__version__ == "1.0.0"


def test_package_exports() -> None:
    import fras.api
    import fras.attendance
    import fras.core
    import fras.db
    import fras.evaluation
    import fras.recognition
    import fras.security
    import fras.training

    assert fras.api is not None
    assert fras.attendance is not None
    assert fras.core is not None
    assert fras.db is not None
    assert fras.evaluation is not None
    assert fras.recognition is not None
    assert fras.security is not None
    assert fras.training is not None
