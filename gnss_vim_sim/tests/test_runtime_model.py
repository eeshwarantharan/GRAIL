from gnss_vim_sim.ml.runtime import HeuristicRuntimeModel, RuntimeModel


def test_runtime_model_has_backward_compatible_predict_risk():
    model: RuntimeModel = HeuristicRuntimeModel()
    features = {
        "vdop": 3.0,
        "mean_cn0": 30.0,
        "n_sats": 6.0,
        "phase_locked_frac": 0.6,
        "mean_delay_ns": 4.0,
    }

    score = model.predict_score(features)

    assert 0.0 <= score <= 1.0
    assert model.predict_risk(features) == score
