import detection


def test_version():
    """Check to see that we can get the package version"""
    assert detection.__version__ is not None
