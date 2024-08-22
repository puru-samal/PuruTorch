from ..nn import Module, Identity, ReLU, BatchNorm2D, Conv2D
from ..tensor import Tensor

class ConvBn2D(Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0) -> None:
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = BatchNorm2D(num_features=out_channels)
    
    def forward(self, x : Tensor) -> Tensor:
        out = self.bn(self.conv(x))
        return out


class ResBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=3, padding=1):
        super().__init__()

        ##self.conv1 = 
        
        self.cbn1 = ConvBn2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cbn2 = ConvBn2D(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        
        if stride != 1 or in_channels != out_channels or kernel_size != 1 or padding != 0:
            self.residual =  ConvBn2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=1)
        else:
            self.residual =  Identity()  
        
        self.act = ReLU()
    
    def forward(self, x : Tensor) -> Tensor:
        identity = x

        out = self.act(self.cbn1(x))
        out = self.cbn2(out)
        out = self.residual(identity) + out

        out = self.act(out)
        return out


