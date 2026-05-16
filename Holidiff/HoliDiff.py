    
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import torch.nn as nn
from Holidiff.micro.tek import TEK
from Holidiff.macro.dpm_sampler import DPMSolverSampler
from Holidiff.utils.diffusion_utils import *
from Holidiff.layers.RevIN import RevIN



def cosine_beta_schedule(timesteps, s=5):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class HATEK(nn.Module):
    
    def __init__(self, configs):
        super(HATEK, self).__init__()

        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.diff_steps = configs.diff_steps
        self.stride=configs.stride
        self.patch_len=configs.patch_len
        self.d_model=configs.d_model
        self.e_layers=configs.e_layers
        self.num_heads=configs.num_heads
        self.rmom_n = configs.rmom
        self.n_blocks = configs.n_b
        self.tek = TEK(configs)
            
        self.enc_in = configs.enc_in
        self.batch_size =configs.batch_size
        self.beta_start = 1e-4 # 1e4
        self.beta_end = 1e-1#2e-2
        self.beta_schedule = 'cosine'
        self.v_posterior = 0.0
        self.loss_type = "l1"
        self.set_micro_uncertainty_schedule(None, self.beta_schedule, self.diff_steps, self.beta_start, self.beta_end)
        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5
        self.nn = self.tek
        self.sampler = DPMSolverSampler(configs,self.nn, self.device,self.alphas_cumprod,self.betas.device)
        self.nda_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
        self.reset_diagnostics()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        if self.training:
            return self.forward_micro_generation_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask)
        else:
            return self.forward_consensus_inference(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times)


    def forward_micro_generation_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        #print(np.shape(x_enc),np.shape(x_mark_enc)) # (B,L,N)
        x = x_dec[:, -self.configs.pred_len:, :].permute(0, 2, 1)
        if x.shape[-1] < self.patch_len:
            pad_len = self.patch_len - x.shape[-1]
            x = F.pad(x, (0, pad_len), mode='replicate')
        f_dim = -1 if self.configs.features in ['MS'] else 0
        x=x[:,f_dim:,:]
        cond_ts = x_enc#(B,L,N)
        if self.configs.new_norm:
            cond_ts = self.nda_layer(cond_ts,'norm')
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L)
            lenth = np.shape(x)[1] 
            mean_ = torch.mean(x[:,-lenth:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x, dim=1).unsqueeze(1))
            x = (x-mean_.repeat(1,lenth,1))/(std_.repeat(1,lenth,1)+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            target_len = self.configs.pred_len
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.generate_micro_realization(x_start=x, t=t, noise=noise)
            model_out= self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out=torch.reshape(model_out,(B,N,target_len))
            model_out = model_out.permute(0,2,1) #(B,TARGET L,N)
            model_out=self.nda_layer(model_out,'denorm')
            model_out = model_out.permute(0,2,1)  #(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        else:
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L) 
            mean_ = torch.mean(cond_ts, dim=-1,keepdims=True)
            std_ = torch.std(cond_ts, dim=-1,keepdims=True)
            cond_ts = (cond_ts-mean_)/(std_+0.00001)
            x = (x-mean_)/(std_+0.00001)
            B = np.shape(x)[0]
            N = self.configs.enc_in
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            target_len = self.configs.pred_len
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            #print(t,t.shape)
            noise = torch.randn_like(x)
            x_k = self.generate_micro_realization(x_start=x, t=t, noise=noise)
            model_out = self.nn(x_k, t, cond_ts,x_mark_enc)
            model_out = torch.reshape(model_out,(B,N,target_len))
            model_out = model_out*(std_+0.00001) + mean_#(B,N,TARGET L)
            weight_tmp = self.sqrt_one_minus_alphas_cumprod[t].reshape(model_out.shape[0],model_out.shape[1],1)
        return model_out,weight_tmp

    def forward_consensus_inference(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        x_future = x_dec[:,-self.configs.pred_len:,:].permute(0,2,1)
        x_past = x_enc.permute(0,2,1)     
        f_dim = -1 if self.configs.features in ['MS'] else 0
        batchs, nF, nL = np.shape(x_past)[0], self.enc_in, self.pred_len
        batchs = batchs*self.enc_in
        if self.configs.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        all_outs = []
        B = np.shape(x_past)[0]
        N = self.configs.enc_in
        if self.configs.new_norm:
            x_past = x_past.permute(0,2,1) #(B,L,N)
            x_past = self.nda_layer(x_past,'norm')
            x_past = x_past.permute(0,2,1) #(B,N,L)
        else:
            mean_ = torch.mean(x_past, dim=-1,keepdims=True)
            std_ = torch.std(x_past, dim=-1,keepdims=True)
            x_past = (x_past-mean_)/(std_+0.00001)
        x_past = torch.reshape(x_past,(B*N,-1))
        x_past = x_past.to(self.betas.device)
        for i in range(sample_times):
            start_code = torch.randn((batchs, nL), device=self.betas.device)
            diff_samples ,_= self.sampler.sample(S=self.configs.s_steps,
                                             conditioning=x_past,
                                             x_mark_enc=x_mark_enc,
                                             batch_size=batchs,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
            diff_samples=torch.reshape(diff_samples,(B,N,-1))      
            if self.configs.new_norm:                       
                diff_samples = diff_samples.permute(0,2,1).to(self.device) #(B,TARGET L,N)
                diff_samples = self.nda_layer(diff_samples,'denorm')
            else:
                diff_samples = diff_samples.to(self.device)*(std_+0.00001) + mean_
                diff_samples = diff_samples.permute(0,2,1)
            all_outs.append(diff_samples)
        all_outs = torch.stack(all_outs, dim=0)
        if self.configs.use_mom and sample_times>1:
                outs = self.extract_consensus(all_outs)
        else:
                outs = all_outs.mean(0)
        
        return outs,all_outs.permute(1,0,2,3)

    def set_micro_uncertainty_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  

        betas = cosine_beta_schedule(diff_steps,self.configs.coss)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))


        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all() 

    def generate_micro_realization(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def _consensus_mean(self, seq):
        return torch.sum(seq, dim=0) / seq.size(0)

    def reset_diagnostics(self):
        self._diag_block_var = []
        self._diag_inter_dev = []
        self._diag_median_bias = []
        self._diag_block_var_map = []
        self._diag_inter_dev_map = []
        self._diag_mean_pred = []
        self._diag_median_pred = []

    def _consensus_reduce(self, tensor):
        if self.n_blocks > tensor.size(0):
            self.n_blocks = int(torch.ceil(tensor.size(0) / 2))

        indic = torch.randperm(tensor.size(0))
        tensor = tensor[indic]
        block_size = tensor.size(0) // self.n_blocks

        means = []
        block_var_maps = []
        block_var_scalars = []
        for i in range(self.n_blocks):
            start_index = i * block_size
            end_index = start_index + block_size if (i + 1) < self.n_blocks else tensor.size(0)
            block = tensor[start_index:end_index]
            block_mean = self._consensus_mean(block)
            means.append(block_mean)
            block_var = torch.var(block, dim=0, unbiased=False)
            block_var_maps.append(block_var)
            block_var_scalars.append(block_var.mean(dim=[1, 2]))

        means = torch.stack(means)
        means_mean = means.mean(dim=0)
        inter_dev_map = torch.mean((means - means_mean.unsqueeze(0)) ** 2, dim=0)
        median_pred = torch.median(means, dim=0)[0]
        median_bias = torch.abs(median_pred - means_mean).mean(dim=[1, 2])

        block_var_map_mean = torch.stack(block_var_maps, dim=0).mean(dim=0)
        block_var_scalar_mean = torch.stack(block_var_scalars, dim=0).mean(dim=0)
        inter_dev_scalar = inter_dev_map.mean(dim=[1, 2])

        self._diag_block_var.append(block_var_scalar_mean.detach().cpu())
        self._diag_inter_dev.append(inter_dev_scalar.detach().cpu())
        self._diag_median_bias.append(median_bias.detach().cpu())
        self._diag_block_var_map.append(block_var_map_mean.detach().cpu())
        self._diag_inter_dev_map.append(inter_dev_map.detach().cpu())
        self._diag_mean_pred.append(means_mean.detach().cpu())
        self._diag_median_pred.append(median_pred.detach().cpu())

        return median_pred

    def extract_consensus(self, outputs):
        results = []
        start_idx = len(self._diag_block_var)
        for _ in range(self.rmom_n):
            shuffled_outputs = outputs[torch.randperm(outputs.size(0))]
            result = self._consensus_reduce(shuffled_outputs)
            results.append(result)
        results = torch.stack(results)

        recent_block_var = self._diag_block_var[start_idx:]
        recent_inter_dev = self._diag_inter_dev[start_idx:]
        recent_median_bias = self._diag_median_bias[start_idx:]
        recent_block_var_map = self._diag_block_var_map[start_idx:]
        recent_inter_dev_map = self._diag_inter_dev_map[start_idx:]
        recent_mean_pred = self._diag_mean_pred[start_idx:]
        recent_median_pred = self._diag_median_pred[start_idx:]

        self._diag_block_var = self._diag_block_var[:start_idx]
        self._diag_inter_dev = self._diag_inter_dev[:start_idx]
        self._diag_median_bias = self._diag_median_bias[:start_idx]
        self._diag_block_var_map = self._diag_block_var_map[:start_idx]
        self._diag_inter_dev_map = self._diag_inter_dev_map[:start_idx]
        self._diag_mean_pred = self._diag_mean_pred[:start_idx]
        self._diag_median_pred = self._diag_median_pred[:start_idx]

        if recent_block_var:
            self._diag_block_var.append(torch.stack(recent_block_var, dim=0).mean(dim=0))
            self._diag_inter_dev.append(torch.stack(recent_inter_dev, dim=0).mean(dim=0))
            self._diag_median_bias.append(torch.stack(recent_median_bias, dim=0).mean(dim=0))
            self._diag_block_var_map.append(torch.stack(recent_block_var_map, dim=0).mean(dim=0))
            self._diag_inter_dev_map.append(torch.stack(recent_inter_dev_map, dim=0).mean(dim=0))
            self._diag_mean_pred.append(torch.stack(recent_mean_pred, dim=0).mean(dim=0))
            self._diag_median_pred.append(torch.stack(recent_median_pred, dim=0).mean(dim=0))
        return self._consensus_mean(results)

