from rainwin import build as build_mod


def test_second_build_skips_while_lock_is_held(tmp_path, monkeypatch):
    # if the lock did not work, build would try to reach ECMWF; make that fail loudly
    monkeypatch.setattr(build_mod, "latest_complete_run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    with build_mod.build_lock(tmp_path) as first:
        assert first is True
        assert build_mod.build(data_dir=tmp_path) is None   # skipped, no network call
    # once released, the lock can be taken again
    with build_mod.build_lock(tmp_path) as again:
        assert again is True
