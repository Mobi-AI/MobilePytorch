import torch

class PixelShuffle(torch.nn.Module):

    '''
    Validated to be equivalent to torch.nn.PixelShuffle in Pytorch 1.4.0
    '''

    def __init__(self, scale, support_dsp):

        super(PixelShuffle, self).__init__()
        self.scale=scale
        self.support_dsp = support_dsp

    def forward(self, x):

        '''
        Restrictions:
        (1) Hexagon AIP does not support 5D tensors (as warned by ``snpe-onnx-to-dlc``).
        (2) Qualcomm Kryo CPUs and Ardeno GPUs only support 1D-5D transpose op.
        '''

        y=x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC//(self.scale*self.scale), iH*self.scale, iW*self.scale

        if self.support_dsp:

            y = torch.split(y, self.scale*self.scale, 1)
            y = [torch.split(sub, self.scale, 1) for sub in y]
            y = [[subsub.permute(0, 2, 3, 1).reshape(B, iH, oW, 1) for subsub in sub] for sub in y]
            y = [torch.cat(sub,3) for sub in y]
            y = [sub.permute(0, 1, 3, 2).reshape(B, 1, oH, oW) for sub in y]
            y = torch.cat(y, 1)

        else:
            
            y = y.contiguous().view(B*oC, self.scale, self.scale, iH, iW)
            y = y.permute(0, 3, 1, 4, 2)
            y = y.contiguous().view(B, oC, oH, oW)

        return y
