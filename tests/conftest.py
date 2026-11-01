import pytest

from app.core.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()
