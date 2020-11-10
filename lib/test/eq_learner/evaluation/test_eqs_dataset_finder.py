import pytest
from eq_learner.evaluation import eqs_dataset_finder
import numpy as np

@pytest.fixture(scope="module")
def mini_dataset(module_mocker):
    dataset = module_mocker.MagicMock()
    import torch
    inp = torch.tensor([[ 4.3097e+00,  4.9382e+00,  5.7547e+00,  6.8324e+00,  8.2727e+00,
          1.0212e+01,  1.2825e+01,  1.6310e+01,  2.0859e+01,  2.6577e+01,
          3.3379e+01,  4.0865e+01,  4.8264e+01,  5.4508e+01,  5.8495e+01,
          5.9449e+01,  5.7222e+01,  5.2353e+01,  4.5854e+01,  3.8848e+01,
          3.2247e+01,  2.6605e+01,  2.2123e+01,  1.8759e+01,  1.6346e+01,
          1.4681e+01,  1.3574e+01,  1.2873e+01,  1.2461e+01,  1.2251e+01],
        [-6.8025e+00, -4.6068e+00, -3.2634e+00, -2.2640e+00, -1.4508e+00,
         -7.5499e-01, -1.4062e-01,  4.1305e-01,  9.1910e-01,  1.3863e+00,
          1.8209e+00,  2.2274e+00,  2.6095e+00,  2.9701e+00,  3.3113e+00,
          3.6353e+00,  3.9435e+00,  4.2375e+00,  4.5185e+00,  4.7875e+00,
          5.0455e+00,  5.2933e+00,  5.5316e+00,  5.7612e+00,  5.9826e+00,
          6.1965e+00,  6.4031e+00,  6.6031e+00,  6.7969e+00,  6.9847e+00]],  dtype=torch.float64)
    out = torch.tensor([[12,  4,  5,  3,  5, 16,  8,  1,  6,  9,  3,  5, 15,  8,  1,  6,  9, 14,
          6,  9,  3,  5,  2,  5,  1,  6,  7, 16,  9,  2,  5,  1,  6,  7, 15,  9,
          2,  5,  1,  6,  9, 14,  6, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0],
        [12,  4,  5,  1,  7, 16,  6,  9,  4,  5,  1,  7, 16,  9,  1,  7, 15,  9,
          1,  9, 14,  6, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0]], dtype=torch.int32)
    dataset.tensors = [inp,out]
    return dataset


def test_normalize(training_dataset_complete):
    num_expression = training_dataset_complete.tensors[0][0].numpy()
    eqs_dataset_finder.normalize(num_expression)

def test_convert_dataset_to_neural_network(mini_dataset):
    interpolation = np.arange(0.1,3,0.1)
    extrapolation = np.arange(3,6,0.1)
    eqs_dataset_finder.convert_dataset_to_neural_network(mini_dataset,interpolation,extrapolation, normalize=True)
