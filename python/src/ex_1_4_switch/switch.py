import numpy as np

_cause_is = "purple"
_cause_probability = 0.5  # If this is not 0.5, then we need no intervention to solve problem
_effect_probability = 0.65


def sample_switch(n_samples, intervene_blue=None, intervene_purple=None):
    if "b" in _cause_is.lower():
        return _sample_switch(n_samples=n_samples, intervene_cause=intervene_blue, intervene_effect=intervene_purple)
    elif "p" in _cause_is.lower():
        return np.flip(
            _sample_switch(n_samples=n_samples, intervene_cause=intervene_purple, intervene_effect=intervene_blue),
            axis=1
        )
    return np.array([
        _sample_cause(n_samples=n_samples),
        _sample_cause(n_samples=n_samples)
    ]).T


def _sample_cause(n_samples: int):
    return np.random.choice([1, 0], size=n_samples, replace=True, p=[_cause_probability, 1 - _cause_probability])


def _sample_effect(causes: np.ndarray):
    n_samples = causes.shape[0]
    flips = np.random.choice([1, 0], size=n_samples, replace=True, p=[1 - _effect_probability, _effect_probability])
    return causes * (1 - flips) + (1 - causes) * flips


def _sample_switch(n_samples, intervene_cause=None, intervene_effect=None):
    # Cause
    if intervene_cause is not None:
        causes = np.ones((n_samples,), dtype=np.int) * intervene_cause
    else:
        causes = _sample_cause(n_samples=n_samples)

    # Effect
    if intervene_effect is not None:
        effects = np.ones((n_samples,), dtype=np.int) * intervene_effect
    else:
        effects = _sample_effect(causes=causes)

    # Combined data
    data = np.stack((causes, effects), axis=1)

    return data
