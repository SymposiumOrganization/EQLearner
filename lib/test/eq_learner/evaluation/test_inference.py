from eq_learner.evaluation import inference

def test_traslate_sentence_from_numbers(training_dataset, model, device):
    num_expression = training_dataset.tensors[0][0].numpy()
    trg_indexes, attention, xxx_n = inference.traslate_sentence_from_numbers(num_expression, model, device, max_len = 66)

