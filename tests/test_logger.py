"""Unit tests for the experiment logger, hash determinism, and non-overwriting protection."""
import os
import shutil
import pytest

from src.experiments.logger import compute_config_hash, ExperimentLogger


@pytest.fixture
def temp_output_dir(tmp_path: os.PathLike) -> str:
    """Fixture providing temporary output directory."""
    path = str(tmp_path / "test_logs")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_config_hash_determinism() -> None:
    """Verify that equivalent configs produce identical hashes regardless of key insertion order."""
    cfg1 = {"sim": {"freq": 240, "gui": False}, "seed": 42}
    cfg2 = {"seed": 42, "sim": {"gui": False, "freq": 240}}
    cfg3 = {"seed": 43, "sim": {"freq": 240, "gui": False}}

    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    hash3 = compute_config_hash(cfg3)

    assert hash1 == hash2, "Expected identical hashes for semantically identical dictionaries"
    assert hash1 != hash3, "Expected different hashes for different parameter values"
    assert len(hash1) == 8


def test_logger_saves_and_prevents_silent_overwrite(temp_output_dir: str) -> None:
    """Verify that logger writes files and increments version suffixes to avoid silent overwrites."""
    cfg = {"param": "value", "seed": 1}

    # Run 1
    logger1 = ExperimentLogger(
        scenario_name="test_scen",
        controller_name="naive",
        seed=1,
        config=cfg,
        output_dir=temp_output_dir,
        allow_overwrite=False,
    )
    logger1.log_step(
        timestamp=0.0,
        step=0,
        drone_id=0,
        pos=[0, 0, 1],
        target_pos=[0, 0, 1],
        rpy=[0, 0, 0],
        vel=[0, 0, 0],
        rpms=[14000, 14000, 14000, 14000],
    )
    csv1, meta1 = logger1.save(summary_metrics={"metric": 1.0})

    assert os.path.exists(csv1)
    assert os.path.exists(meta1)

    # Run 2 with identical settings without allow_overwrite
    logger2 = ExperimentLogger(
        scenario_name="test_scen",
        controller_name="naive",
        seed=1,
        config=cfg,
        output_dir=temp_output_dir,
        allow_overwrite=False,
    )
    csv2, meta2 = logger2.save(summary_metrics={"metric": 2.0})

    # Logger 2 must NOT have overwritten logger 1's files
    assert csv1 != csv2
    assert meta1 != meta2
    assert "_v1.csv" in csv2
    assert os.path.exists(csv1)
    assert os.path.exists(csv2)
