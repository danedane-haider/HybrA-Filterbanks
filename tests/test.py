import importlib.metadata
import hybra as m

def test_version():
    assert importlib.metadata.version("hybra") == m.__version__