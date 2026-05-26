import os
import torch
from Holidiff.HoliDiff import HATEK
from Holidiff.baselines.diffusion_ts_adapter import DiffusionTSAdapter
from Holidiff.baselines.mgtsd_adapter import MGTSDAdapter
from Holidiff.baselines.tsdiff_adapter import TSDiffAdapter


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'HATEK': HATEK,
            'DiffusionTS': DiffusionTSAdapter,
            'TSDiff': TSDiffAdapter,
            'MGTSD': MGTSDAdapter,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                torch.cuda.set_device(device)
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
