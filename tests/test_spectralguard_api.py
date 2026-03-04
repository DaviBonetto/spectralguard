import pytest

from spectralguard import SpectralGuardDetector, monitor, set_default_detector


def test_package_monitor_contract():
    set_default_detector(SpectralGuardDetector(threshold=0.5))
    is_safe, score = monitor("safe prompt", [0.99, 0.98, 0.97])
    assert isinstance(is_safe, bool)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_monitor_rejects_invalid_inputs():
    detector = SpectralGuardDetector()
    with pytest.raises(ValueError):
        detector.monitor(123, [0.9, 0.8])
    with pytest.raises(ValueError):
        detector.monitor("x", {"invalid": [0.9, 0.8]})
    with pytest.raises(ValueError):
        detector.monitor("x", [])
    with pytest.raises(ValueError):
        detector.monitor("x", {"rho_layers": [0.9, float("nan")]})


def test_monitor_is_deterministic_for_same_input():
    detector = SpectralGuardDetector(threshold=0.5)
    hs = [0.96, 0.95, 0.94]
    out_1 = detector.monitor("prompt", hs)
    out_2 = detector.monitor("prompt", hs)
    assert out_1[0] == out_2[0]
    assert out_1[1] == pytest.approx(out_2[1], rel=0.0, abs=0.0)


def test_thresholding_behavior():
    risky = [0.15, 0.10, 0.20]
    safe = [0.99, 0.98, 0.97]

    strict = SpectralGuardDetector(threshold=0.2)
    assert strict.monitor("prompt", risky)[0] is False
    assert strict.monitor("prompt", safe)[0] is True

    permissive = SpectralGuardDetector(threshold=0.9)
    assert permissive.monitor("prompt", risky)[0] is True


def test_monitor_accepts_dict_rho_key():
    detector = SpectralGuardDetector(threshold=0.5)
    is_safe, score = detector.monitor("prompt", {"rho": [0.99, 0.98, 0.97]})
    assert isinstance(is_safe, bool)
    assert isinstance(score, float)


def test_set_default_detector_type_guard():
    with pytest.raises(ValueError):
        set_default_detector("invalid")
