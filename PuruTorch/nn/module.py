from ..tensor import Parameter
from typing import List

class Module:
    ''' General Module superclass'''
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        '''
        Returns all model parameters in a list. Iterates over each item in self.__dict__,
        and returns every Parameter object.
        '''
        params = []
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                params += param.parameters()
            elif isinstance(param, Parameter):
                params.append(param)
        return params

    def train(self) -> None:
        ''' Sets module's mode to train, which influences layers like Dropout'''
        self.mode = 'train'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.train()
    
    def eval(self) -> None:
        ''' Sets module's mode to eval, which influences layers like Dropout'''
        self.mode = 'eval'
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.eval()
