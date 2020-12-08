from collections import defaultdict

class PolynomialCompoenents():
    def __init__(self):
        self.combinations = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.symbol = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def elements(self, obj=0):
        if obj == 0:
            return self._num_elements(self.combinations)
        else:
            return self._num_elements(obj)

    @classmethod
    def _num_elements(cls,x):
        if isinstance(x, dict):
            return sum([cls._num_elements(_x) for _x in x.values()])
        else: return len(x)
        