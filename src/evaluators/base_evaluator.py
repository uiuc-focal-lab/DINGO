from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
    
    @abstractmethod
    def evaluate_answer(self, gen_output):
        pass
    