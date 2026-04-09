"""测试 SimDiff + T/S 层架构"""
import torch
import sys
sys.path.insert(0, '/home/yanyijin/research/MSST-CL/models')

# 模拟 configs
class Config:
    seq_len = 96
    pred_len = 96
    patch_len = 16
    stride = 8
    d_model = 128
    num_heads = 4
    e_layers = 2
    dropout = 0.1
    skip_dropout = 0.1
    d_ff = 256
    enc_in = 7
    batch_size = 8
    diff_steps = 100
    s_steps = 5
    rmom = 20
    n_b = 5
    coss = 0.008
    new_norm = True
    features = 'M'

config = Config()
# DPMSolverSampler 需要额外的属性
config.lower_order_final = "true"
config.method = "multistep"
config.order = 2

print("=" * 50)
print("Step 1: 导入模块")
print("=" * 50)
try:
    from fourier.ptld_model import Model, PatchUVIT, FormerBone
    print("✓ 导入成功!")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 2: 创建模型")
print("=" * 50)
try:
    model = Model(config)
    print(f"✓ 模型创建成功!")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 3: 测试 forward_train")
print("=" * 50)
try:
    model.train()
    # 创建测试数据（在CPU上）
    B, L, N = 4, 96, 7
    x_enc = torch.randn(B, L, N)
    x_mark_enc = torch.randn(B, L, 4)
    x_dec = torch.randn(B, L + 48, N)
    x_mark_dec = torch.randn(B, L + 48, 4)

    # 强制模型在CPU上
    device = torch.device('cpu')
    model = model.to(device)
    config.device = 'cpu'
    model.device = 'cpu'

    print(f"输入 x_enc shape: {x_enc.shape}")
    print(f"输入 x_dec shape: {x_dec.shape}")
    
    output, weight = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"✓ forward_train 成功!")
    print(f"  输出 shape: {output.shape}")
    print(f"  weight shape: {weight.shape}")
except Exception as e:
    print(f"✗ forward_train 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 4: 验证输出维度")
print("=" * 50)
# forward_train 输出是 (B, N, pred_len)
expected_shape = (B, N, config.pred_len)
if output.shape == expected_shape:
    print(f"✓ 输出维度正确: {output.shape}")
else:
    print(f"✗ 输出维度错误: 期望 {expected_shape}, 实际 {output.shape}")

print("\n" + "=" * 50)
print("Step 5: 测试 loss 计算")
print("=" * 50)
try:
    # 创建 target，形状与 output 一致 (B, N, pred_len)
    target = torch.randn(B, N, config.pred_len)
    loss = torch.nn.functional.l1_loss(output, target)
    print(f"✓ Loss 计算成功: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Loss 计算失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("所有测试通过!")
print("=" * 50)
