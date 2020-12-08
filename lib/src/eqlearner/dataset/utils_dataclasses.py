from dataclasses import dataclass
from typing import List, Tuple
import sympy

@dataclass
class NumElements:
    """Set the types of functions"""
    number_of_terms: int
    number_of_symbols: int

@dataclass
class Constants:
    """Set constants up"""
    constant_intervals_ext: List[Tuple]
    constant_intervals_int: List[Tuple]

@dataclass
class Function:
    elem_without_constant: str
    elem_with_constant: sympy
