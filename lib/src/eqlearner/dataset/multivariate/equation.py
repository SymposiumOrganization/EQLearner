from ..processing.tokenization import tokenize

class Equation():
    def __init__(self, name, token, eq_instance: str, points):
        self.name = name
        self.token = token
        self.eq_instances = [eq_instance]
        self.eq_points = [points]
        
    
    def add_eq(self,eq_instance,points):
        if eq_instance not in self.eq_instances:
            self.eq_instances.append(eq_instance)
            self.eq_points.append(points)

    def remove_eq(self, index = -1):
        self.eq_instances.pop(index)
        self.eq_points.pop(index)
    
    def repetitions(self):
        return len(self.eq_instances) 
