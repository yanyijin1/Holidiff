import copy
import torch
import torch.nn as nn

from .gaussian_diffusion import Diffusion_TS


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.total_len = int(getattr(configs, 'diffusion_ts_seq_length', self.seq_len + self.pred_len))
        self.feature_size = int(getattr(configs, 'diffusion_ts_feature_size', self.enc_in))
        self.clip_denoised = bool(getattr(configs, 'diffusion_ts_clip_denoised', False))
        self.use_ema = bool(getattr(configs, 'diffusion_ts_use_ema', True))
        self.use_ema_sampling = bool(getattr(configs, 'diffusion_ts_use_ema_sampling', self.use_ema))
        self.ema_decay = float(getattr(configs, 'diffusion_ts_ema_decay', 0.995))
        self.ema_update_every = max(1, int(getattr(configs, 'diffusion_ts_ema_update_every', 10)))
        self._ema_step = 0

        self.diffusion = Diffusion_TS(
            seq_length=self.total_len,
            feature_size=self.feature_size,
            n_layer_enc=int(getattr(configs, 'diffusion_ts_n_layer_enc', 3)),
            n_layer_dec=int(getattr(configs, 'diffusion_ts_n_layer_dec', 2)),
            d_model=int(getattr(configs, 'diffusion_ts_d_model', getattr(configs, 'd_model', 64))),
            timesteps=int(getattr(configs, 'diffusion_ts_timesteps', 500)),
            sampling_timesteps=int(getattr(configs, 'diffusion_ts_sampling_timesteps', 200)),
            loss_type=str(getattr(configs, 'diffusion_ts_loss_type', 'l1')),
            beta_schedule=str(getattr(configs, 'diffusion_ts_beta_schedule', 'cosine')),
            n_heads=int(getattr(configs, 'diffusion_ts_n_heads', 4)),
            mlp_hidden_times=int(getattr(configs, 'diffusion_ts_mlp_hidden_times', 4)),
            eta=float(getattr(configs, 'diffusion_ts_eta', 0.0)),
            attn_pd=float(getattr(configs, 'diffusion_ts_attn_pd', 0.0)),
            resid_pd=float(getattr(configs, 'diffusion_ts_resid_pd', 0.0)),
            use_ff=bool(getattr(configs, 'diffusion_ts_use_ff', True)),
            reg_weight=getattr(configs, 'diffusion_ts_reg_weight', None),
        )

        self._ema_model_holder = [copy.deepcopy(self.diffusion)] if self.use_ema else [None]
        if self.use_ema:
            self._reset_ema_model()

    def _reset_ema_model(self):
        if not self.use_ema:
            return
        ema_model = self._ema_model_holder[0]
        if ema_model is None:
            ema_model = copy.deepcopy(self.diffusion)
            self._ema_model_holder[0] = ema_model
        ema_model.load_state_dict(self.diffusion.state_dict())
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    def _ema_model(self, reference_tensor=None):
        if not self.use_ema or not self.use_ema_sampling:
            return self.diffusion
        ema_model = self._ema_model_holder[0]
        if ema_model is None:
            self._reset_ema_model()
            ema_model = self._ema_model_holder[0]
        if reference_tensor is not None:
            ema_model.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        ema_model.eval()
        return ema_model

    def get_extra_state(self):
        if not self.use_ema or self._ema_model_holder[0] is None:
            return {'ema_step': int(self._ema_step)}
        ema_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self._ema_model_holder[0].state_dict().items()
        }
        return {
            'ema_step': int(self._ema_step),
            'ema_model_state': ema_state,
        }

    def set_extra_state(self, state):
        state = state or {}
        self._ema_step = int(state.get('ema_step', 0))
        if not self.use_ema:
            return
        ema_state = state.get('ema_model_state')
        if ema_state:
            if self._ema_model_holder[0] is None:
                self._ema_model_holder[0] = copy.deepcopy(self.diffusion)
            self._ema_model_holder[0].load_state_dict(ema_state, strict=False)
            self._ema_model_holder[0].eval()
            for parameter in self._ema_model_holder[0].parameters():
                parameter.requires_grad_(False)
        else:
            self._reset_ema_model()

    @torch.no_grad()
    def after_optimizer_step(self):
        if not self.use_ema:
            return
        self._ema_step += 1
        if self._ema_step % self.ema_update_every != 0:
            return
        reference_parameter = next(self.diffusion.parameters())
        ema_model = self._ema_model_holder[0]
        if ema_model is None:
            self._reset_ema_model()
            ema_model = self._ema_model_holder[0]
        ema_model.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        current_state = self.diffusion.state_dict()
        ema_state = ema_model.state_dict()
        for name, current_tensor in current_state.items():
            ema_tensor = ema_state[name]
            source_tensor = current_tensor.detach().to(device=ema_tensor.device, dtype=ema_tensor.dtype)
            if torch.is_floating_point(ema_tensor):
                ema_tensor.mul_(self.ema_decay).add_(source_tensor, alpha=1.0 - self.ema_decay)
            else:
                ema_tensor.copy_(source_tensor)

    def _full_sequence(self, batch_x, batch_y=None):
        if batch_y is None:
            future = torch.zeros(batch_x.size(0), self.pred_len, batch_x.size(-1), device=batch_x.device, dtype=batch_x.dtype)
        else:
            future = batch_y[:, -self.pred_len:, :].to(batch_x.device)
        return torch.cat([batch_x, future], dim=1)

    def training_loss(self, batch_x, batch_y, batch_y_mask=None, **kwargs):
        full_series = self._full_sequence(batch_x, batch_y)
        return self.diffusion(full_series)

    def _infill_kwargs(self):
        return {
            'coef': float(getattr(self.configs, 'diffusion_ts_langevin_coef', 0.01)),
            'learning_rate': float(getattr(self.configs, 'diffusion_ts_langevin_lr', 0.05)),
            'coef_': float(getattr(self.configs, 'diffusion_ts_langevin_coef_inner', 0.0)),
        }

    @torch.no_grad()
    def _sample_once(self, batch_x):
        sample_model = self._ema_model(batch_x)
        target = torch.zeros(batch_x.size(0), self.total_len, batch_x.size(-1), device=batch_x.device, dtype=batch_x.dtype)
        target[:, :self.seq_len, :] = batch_x
        partial_mask = torch.zeros_like(target, dtype=torch.bool)
        partial_mask[:, :self.seq_len, :] = True
        shape = (batch_x.size(0), self.total_len, self.feature_size)

        if sample_model.fast_sampling:
            samples = sample_model.fast_sample_infill(
                shape,
                target,
                sampling_timesteps=sample_model.sampling_timesteps,
                partial_mask=partial_mask,
                clip_denoised=self.clip_denoised,
                model_kwargs=self._infill_kwargs(),
            )
        else:
            samples = sample_model.sample_infill(
                shape,
                target,
                partial_mask=partial_mask,
                clip_denoised=self.clip_denoised,
                model_kwargs=self._infill_kwargs(),
            )
        return samples[:, -self.pred_len:, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, sample_times=1, **kwargs):
        num_samples = max(1, int(sample_times))
        sample_list = [self._sample_once(x_enc) for _ in range(num_samples)]
        all_samples = torch.stack(sample_list, dim=1)
        outputs = all_samples.mean(dim=1)
        return outputs, all_samples
