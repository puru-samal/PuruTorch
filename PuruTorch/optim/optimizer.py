
class Optimizer():
    """Superclass class for optimizers."""
    def __init__(self, params):
        self.params = list(params)
        self.state = [] 

    def step(self):
        """Called after generating gradients; updates network weights."""
        raise NotImplementedError

    def zero_grad(self):
        """After stepping, This can be called to reset gradients."""
        for param in self.params:
            param.grad = None