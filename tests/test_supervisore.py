"""Il supervisore Webots gira end-to-end sul finto Webots e produce il log con le colonne attese."""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_webots import FakeSupervisor  # noqa: E402


def _load_supervisor():
    path = ROOT / "webots" / "controllers" / "gnn_supervisor" / "gnn_supervisor.py"
    spec = importlib.util.spec_from_file_location("gnn_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supervisore_episodio_completo(tmp_path):
    sup = _load_supervisor()
    log = tmp_path / "webots_log.csv"
    sup.main(["--episodes", "1", "--seed", "1000", "--max-time", "60", "--log", str(log)], supervisor=FakeSupervisor(6))
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 1
    row = rows[0]
    for col in ("success", "contact_events", "min_separation", "shelf_contact_max_s", "wait_time_s", "reverse_events"):
        assert col in row
    assert row["success"] == "1"
    assert row["contact_events"] == "0"
    assert float(row["min_separation"]) >= 0.60
