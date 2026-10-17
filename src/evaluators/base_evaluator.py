from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, dataset, do_cot=False):
        self.dataset = dataset
        self.do_cot = do_cot

    @abstractmethod
    def evaluate_answer(self, gen_output):
        pass
