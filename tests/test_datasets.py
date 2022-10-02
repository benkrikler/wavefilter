from wavefilter import datasets

def test_generate_double_pulse_dataset():
    n_samples = 100
    data, truth = datasets.generate_double_pulse_dataset(n_samples, 30)
    assert len(data) == n_samples
    assert len(truth) == 2
    assert len(truth[0]) == n_samples
