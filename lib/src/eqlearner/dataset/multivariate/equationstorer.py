from typing import Dict, List
from .equation import Equation
from collections import defaultdict


class EquationsStorer():
    def __init__(self,family):
        self.family = family
        self.repetition = defaultdict(set)
        self.eqs = dict()
        self.eqs_drop = set()
        self.multiple_instance_enabled = self.keep_multiple_instances()
    
    def keep_multiple_instances(self):
        #if self.family.constants_enabled:
        return True
        #else:
            #return False

    def add_eq_drop(self, eq_drop):
        self.eqs_drop.add(eq_drop)
    
    def add_eq(self, eq_name:str,token, eq_real:str, points):
        if eq_name in self.eqs:
            if self.multiple_instance_enabled:
                current = self.eqs[eq_name].repetitions()
                self.repetition[current].remove(eq_name)
                self.repetition[current + 1].add(eq_name)
                self.eqs[eq_name].add_eq(eq_real,points)
            
        else:
            eq = Equation(eq_name,token,eq_real,points)
            self.repetition[1].add(eq_name)
            self.eqs[eq_name] = eq
                
    def number_of_equations(self):
        return sum([k*len(s) for k,s in self.repetion.items()])



