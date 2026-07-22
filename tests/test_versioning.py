import json

from burner.versioning import resolve_version

CORE = {"params": {"Length": 8.0}, "exit_modules": [], "source_state_sha256": "a" * 64}


def _write_version(out_dir, n, text, core):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"Mini_BTC_H1_v{n}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"Mini_BTC_H1_v{n}.manifest.json").write_text(
        json.dumps(core), encoding="utf-8")


def test_first_burn_is_v1(tmp_path):
    v, reused = resolve_version(tmp_path, "Mini", "BTC", "H1",
                                lambda n: f"content v{n}", CORE)
    assert (v, reused) == (1, False)


def test_identical_rerun_reuses(tmp_path):
    _write_version(tmp_path, 1, "content v1", CORE)
    v, reused = resolve_version(tmp_path, "Mini", "BTC", "H1",
                                lambda n: f"content v{n}", CORE)
    assert (v, reused) == (1, True)


def test_content_change_bumps(tmp_path):
    _write_version(tmp_path, 1, "content v1", CORE)
    v, reused = resolve_version(tmp_path, "Mini", "BTC", "H1",
                                lambda n: f"NEW content v{n}", CORE)
    assert (v, reused) == (2, False)


def test_param_change_bumps_even_if_text_matches(tmp_path):
    _write_version(tmp_path, 1, "content v1", CORE)
    new_core = dict(CORE, params={"Length": 9.0})
    v, reused = resolve_version(tmp_path, "Mini", "BTC", "H1",
                                lambda n: f"content v{n}", new_core)
    assert (v, reused) == (2, False)


def test_old_versions_never_touched(tmp_path):
    _write_version(tmp_path, 1, "content v1", CORE)
    _write_version(tmp_path, 2, "other v2", dict(CORE, source_state_sha256="b" * 64))
    v, reused = resolve_version(tmp_path, "Mini", "BTC", "H1",
                                lambda n: f"third v{n}", CORE)
    assert (v, reused) == (3, False)
    assert (tmp_path / "Mini_BTC_H1_v1.txt").read_text(encoding="utf-8") == "content v1"
