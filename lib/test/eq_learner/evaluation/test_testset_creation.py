from eq_learner.evaluation import testset_creation
import numpy as np
import pytest

@pytest.fixture(scope="module")
def dictionary_creator(training_list):
    dictionary_creator = testset_creation.dictionary_creator(training_list)
    return dictionary_creator

@pytest.mark.parametrize("k",[(10)])
def test_unique_sets_of_tokens(k, training_list):
    res = testset_creation.unique_sets_of_tokens(training_list[1],k)
    proof = set(map(tuple,res))
    assert len(res) == len(proof)


def test_dictionary_creator(training_list):
    res = testset_creation.dictionary_creator(training_list)
    assert len(training_list[0]) >= len(res)

def test_training_set_creation(dictionary_creator):
    res = testset_creation.training_set_creation(dictionary_creator,k=10)
    assert len(res.tensors[0]) == len(res.tensors[1])

def test_generate_training_set_from_dataset_creation(dictionary_creator,DatasetCreator):
    support = np.arange(0.1,3,0.1)
    testset_creation.generate_training_set_from_dataset_creation(dictionary_creator, DatasetCreator, support, number=2)

def test_generate_val_set_from_dataset_creation(dictionary_creator,DatasetCreator):
    support = np.arange(0.1,3,0.1)
    testset_creation.generate_val_set_from_dataset_creation(dictionary_creator, DatasetCreator, support, number=2)