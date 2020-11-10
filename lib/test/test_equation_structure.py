import pytest
from sympy import sin, Symbol, log, exp, zoo
from eq_learner.DatasetCreator.DatasetCreator import DatasetCreator
from eq_learner.DatasetCreator.EquationStructure import EquationStructures
import numpy as np
from sympy import simplify

def test_polynomials():
    x = Symbol('x')
    basis_functions = [x,exp,log,sin] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    eq = EquationStructures(basis_functions)
    eq.polynomial

def test_polynomial_jointer():
    x = Symbol('x')
    candidate = [1,x,0,x,0,0,x]
    vers2 = EquationStructures.polynomial_joiner(candidate,x,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)])
    vers1 = 1 + x + x**3 + x**6
    assert simplify(vers1-vers2) == 0 


def test_composition_creator():
    x = Symbol('x')
    candidate = [1,x,0,x,0,0,x]
    vers2 = EquationStructures.polynomial_joiner(candidate,x,const_interval_ext=[(1,1)], constant_interval_int=[(1,1)])
    vers1 = 1 + x + x**3 + x**6
    assert simplify(vers1-vers2) == 0 