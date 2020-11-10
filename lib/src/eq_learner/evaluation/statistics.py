from collections import Counter
from eq_learner.processing.tokenization import get_string
from eq_learner.evaluation import inference, eqs_dataset_finder, statistics
from eq_learner.evaluation.eqs_dataset_finder import normalize
from eq_learner.processing.tokenization import get_string
from collections import defaultdict
from . import eqs_dataset_finder
import torch
from sympy import lambdify, Symbol
import numpy as np
import functools
from itertools import repeat
import numpy as np
import math 

def count_occurences(texted):
    res = Counter(texted)
    return res

def count_number_occurences(tokens):
    t= list(map(len,tokens))
    res = Counter(t)
    return res

def is_correct_with_symbols(gt,pred):
    if len(gt) != len(pred):
        return False
    return all(gt==pred)

def rmse_calculator(gt,pred, interpolation, extrapolation, normalization=True):
    try:
        gt_y, gt_y_extra = eqs_dataset_finder.compute_Y(gt,interpolation,extrapolation,normalization=normalization)
        pred_y, pred_y_extra = eqs_dataset_finder.compute_Y(pred,interpolation,extrapolation,normalization=normalization)
        
    # x = Symbol('x')
    # gt_string = get_string(gt)
    # pred_string = get_string(pred)
    # function_gt = lambdify(x, gt_string)
    # try:
    #     function_pred = lambdify(x,pred_string)
    # except:
    #     return float('-inf'), float('-inf'), False
    # gt_y = np.array([function_gt(i) for i in interpolation])
    # pred_y = np.array([function_pred(i) for i in interpolation])
    # gt_y_extra = np.array([function_gt(i) for i in extrapolation])
    # pred_y_extra = np.array([function_pred(i) for i in extrapolation])
    # if normalization:
    #     gt_y, scaler = normalize(gt_y)
    #     gt_y_extra, _ = normalize(gt_y_extra,scaler)
    #     pred_y, scaler = normalize(pred_y)
    #     pred_y_extra, _  = normalize(pred_y_extra,scaler)
    
        diff = (gt_y-pred_y)
        diff_extra = (gt_y_extra-pred_y_extra)
        rms = np.sqrt(np.mean(diff**2))
        rms_extra = np.sqrt(np.mean(diff_extra**2))
    # if abs(rms)<0.2 and abs(rms_extra) <0.2:
    #     symbolic_loss = True
    #     rms, symbolic_loss
    # else: 
    #     symbolic_loss = False
    except:
        rms, rms_extra = np.nan, np.nan
    return rms, rms_extra
    
def evaluation_pipeline(numerical_values,eq_instance,interpolation, extrapolation, model,device=torch.device('cuda')):
    num_expression, _ = eqs_dataset_finder.normalize(numerical_values)
    trg_indexes, attention, xxx_n = inference.traslate_sentence_from_numbers(num_expression, 
                                                                         model, device, max_len = 59)
    gt_expression = eqs_dataset_finder.remove_padding(eq_instance)
    pred_expression = trg_indexes
    return statistics.rmse_calculator(gt_expression,pred_expression, interpolation, extrapolation), gt_expression, pred_expression

def dict_creator(train_tensor, interpolation,extrapolation, model,device=torch.device('cuda')):
    res = dict()
    res["rmse"], res["rmse_extra"], res["gt_expressions"], res["pred_expression"] = list(), list(), list(), list()
    for i in range(len(train_tensor.tensors[0])):
        rmse, gt_expressions, pred_expression = statistics.evaluation_pipeline(train_tensor.tensors[0][i].numpy(),
                                            train_tensor.tensors[1][i].numpy(),
                                            interpolation, extrapolation, model)
        res["rmse"].append(rmse[0]),  res["rmse_extra"].append(rmse[1]), 
        res["gt_expressions"].append(get_string(gt_expressions)), res["pred_expression"].append(get_string(pred_expression))
    return res 
                        

def total_rmse_calculator(sol):
    average_rmse = 0
    average_rmse_extra = 0
    counter = 0
    symbolic_precision = 0
    for x in sol:
        if x[0][0] + x[0][1] != float('-inf') and not math.isnan(x[0][0] + x[0][1]):
            average_rmse = x[0][0] + average_rmse
            average_rmse_extra = x[0][1] + average_rmse_extra
            counter = counter + 1
        if x[0][2]:
            symbolic_precision = symbolic_precision + 1
    res = average_rmse/counter
    res_extra = average_rmse_extra/counter
    tot = symbolic_precision/counter
    return res, res_extra, tot