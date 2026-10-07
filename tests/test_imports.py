"""Every module imports under the package layout (catches a stale bare `from data import ...`)."""
import importlib
import pkgutil

import pytest

import scsi_new

MODULES = [m.name for m in pkgutil.walk_packages(scsi_new.__path__, "scsi_new.")]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
