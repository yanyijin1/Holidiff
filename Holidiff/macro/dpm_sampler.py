"""SAMPLING ONLY."""

import torch

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    def __init__(self, args, model, df_device, df_alphas_cumprod, df_betas_device, **kwargs):
        super().__init__()
        self.args = args
        self.diff_steps = args.diff_steps,
        if isinstance(args.lower_order_final, bool):
            self.lower_order_final = args.lower_order_final
        else:
            self.lower_order_final = True if str(args.lower_order_final).lower() == 'true' else False

        self.model = model
        self.df_device = df_device
        self.df_alphas_cumprod = df_alphas_cumprod
        self.df_betas_device = df_betas_device
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.df_device)
        self.register_buffer('alphas_cumprod', to_torch(self.df_alphas_cumprod))

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            target_device = torch.device(self.df_device) if not isinstance(self.df_device, torch.device) else self.df_device
            if attr.device != target_device:
                attr = attr.to(target_device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               x_mark_enc=None,
               raw_history=None,
               physical_injection=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        _, L = shape
        size = (batch_size, L)
        device = self.df_betas_device
        img = torch.randn(size, device=device) if x_T is None else x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        model_fn = model_wrapper(
            self.diff_steps,
            lambda x, t, c, m, **extra_kwargs: self.model.forward(x, t, c, m, **extra_kwargs),
            ns,
            model_type="x_start",
            guidance_type="classifier-free",
            condition=conditioning,
            x_mark_enc=x_mark_enc,
            model_kwargs={},
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(
            img,
            steps=S,
            skip_type=self.args.skip_type,
            method=self.args.method,
            order=self.args.order,
            lower_order_final=self.lower_order_final,
        )

        return x.to(device), None
