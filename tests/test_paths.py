import pytest

from scsi_new import paths


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SCSI_CLUSTER", "nyu")
    monkeypatch.delenv("SCSI_RUNS_DIR", raising=False)
    assert paths.runs_dir() == paths.PROFILES["nyu"]["runs_dir"]
    monkeypatch.setenv("SCSI_RUNS_DIR", "/somewhere")
    assert paths.runs_dir() == "/somewhere"
    monkeypatch.setenv("SCSI_CLUSTER", "rusty")
    monkeypatch.delenv("SCSI_IGG_DATA_ROOT", raising=False)
    assert paths.igg_data_root().startswith("/mnt/ceph")


def test_bad_cluster(monkeypatch):
    monkeypatch.setenv("SCSI_CLUSTER", "mars")
    with pytest.raises(ValueError):
        paths.cluster()


def test_torchvision_dir_per_cluster_and_experiment(monkeypatch, tmp_path):
    monkeypatch.delenv("SCSI_DATA_DIR", raising=False)
    monkeypatch.setenv("SCSI_CLUSTER", "nyu")
    f = tmp_path / "cryoet_mnist3d" / "data.py"
    assert paths.torchvision_data_dir(str(f)) == "/scratch/cm6627/scsi_data/cryoet_mnist3d"
    monkeypatch.setenv("SCSI_CLUSTER", "rusty")
    assert paths.torchvision_data_dir(str(f)) == "/mnt/ceph/users/cmiyamoto/scsi_data/cryoet_mnist3d"
    monkeypatch.setenv("SCSI_DATA_DIR", "/d")
    assert paths.torchvision_data_dir(str(f)) == "/d"
