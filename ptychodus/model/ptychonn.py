from __future__ import annotations
import logging

try:
    import torch
except ImportError:
    torch = None

from .observer import Observable, Observer
from .reconstructor import Reconstructor
from .settings import SettingsRegistry, SettingsGroup


logger = logging.getLogger(__name__)


# class ReconModel(torch.nn.Module):
# 	def __init__(self, nconv: int = 32):
# 		super().__init__()
#
# 		self.encoder = torch.nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
# 				torch.nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.MaxPool2d((2,2)),
#
# 				torch.nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.MaxPool2d((2,2)),
#
# 				torch.nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.MaxPool2d((2,2)),
# 				)
#
# 		self.decoder1 = torch.nn.Sequential(
# 				torch.nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
# 				torch.nn.Sigmoid() #Amplitude model
# 				)
#
# 		self.decoder2 = torch.nn.Sequential(
# 				torch.nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
# 				torch.nn.ReLU(),
# 				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
#
# 				torch.nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
# 				torch.nn.Tanh() #Phase model
# 				)
#
# 	def forward(self,x):
# 		x1 = self.encoder(x)
# 		amp = self.decoder1(x1)
# 		ph = self.decoder2(x1)
#
# 		#Restore -pi to pi range
# 		ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi
#
# 		return amp,ph


class PtychoNNReconstructor(Reconstructor):
    @property
    def name(self) -> str:
        return 'PNN'

    @property
    def backendName(self) -> str:
        return 'PtychoNN'

    def reconstruct(self) -> int:
        # PyTorch/Keras
        # 1) Load best_model.pth
        # 2) Predict
        # 3) Stitch
        return 0 # TODO


class PtychoNNBackend:
    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool = False) -> PtychoNNBackend:
        core = cls(settingsRegistry)

        if torch or isDeveloperModeEnabled:
            core.reconstructorList.append(PtychoNNReconstructor())
        else:
            logger.info('torch not found.')

        return core

