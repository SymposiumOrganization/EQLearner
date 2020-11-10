import itertools
import sympy
from . import utils
import numpy as np
import random
from collections import defaultdict




class EquationTracker():
    def __init__(self, list_of_eq):
        self.list_of_eq = list_of_eq
        #self.polynomial = self.order_assigner(self._polynomial(basis_function))
        #self.n_terms = n_terms(basis_function)
        #self.compositions = compositions(basis_function)

    def get_equation(self, drop: int = 0):
        """Drop is used to specify which level should we get rid of"""
        path = []
        curr = self.list_of_eq
        #total_keys = list(self.list_of_eq.keys())
        # if total_keys:
        #     k = random.choice(total_keys)
        #     path.append(k)
        # else:
        #     raise ValueError

        # curr = self.list_of_eq[k]
        while type(curr) == defaultdict:
            res = random.choice(list(curr.keys()))
            path.append(res)
            curr = curr[res]
        if len(curr)>1:
            tmp = random.randint(0,len(curr)-1)
            path.append(tmp)
            res = curr[tmp]
        else:
            res = curr[0]
        self._drop_element(self.list_of_eq, path, drop)
        return res

    @staticmethod
    def _drop_element(li, choice, drop):
        if drop >= 0:
            target = li
            for i in choice[:(drop)]:
                target = target[i]
            del target[choice[drop]]
        return

    @staticmethod
    def _drop_element(li, choice, drop):
        if drop >= 0:
            target = li
            for i in choice[:(drop)]:
                target = target[i]
            del target[choice[drop]]
        return
            



    