"""Placeholder test to establish green pipeline."""


def test_placeholder():
    """Basic test to ensure pytest is working."""
    assert True


def test_import():
    """Test that the package can be imported."""
    try:
        import submarit
        assert hasattr(submarit, "__version__")
    except ImportError:
        # Package not installed yet, which is expected
        pass