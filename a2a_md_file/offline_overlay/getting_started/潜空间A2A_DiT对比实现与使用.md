# GR00T 潜空间 A2A：MLP 与时间序列 DiT 对比实现

本文说明仓库中新增的时间序列 latent-token DiT 后端，以及如何在不影响原有
4-block AdaLN-MLP A2A 的前提下训练和对比两种 Action Head。

## 1. 实现边界

本次修改没有替换原有实现，也没有修改原始 `Gr00tN1d7`：

- `Gr00tN1d7`：原始 GR00T N1.7，保持不变；
- `Gr00tN1d7A2A + a2a_flow_backbone="mlp"`：原有论文核心迁移，仍是默认值；
- `Gr00tN1d7A2A + a2a_flow_backbone="dit"`：新增时间序列 token DiT 对照组。

旧 A2A 配置没有 `a2a_flow_backbone` 时会采用默认值 `mlp`，原 MLP checkpoint
的模块名和参数路径没有改变。MLP 与 DiT 的 action-head 图不同，因此代码会拒绝把
完整 MLP A2A checkpoint 静默加载成 DiT，反方向也一样。做公平对比时，两组应从
同一个原始 `Gr00tN1d7` base checkpoint 开始，或分别使用各自架构的 AE warmup
checkpoint。

DiT 是 GR00T 上的研究扩展，不是 A2A 论文已经报告的网络。论文核心对照仍是 MLP；
选择 DiT 时必须显式关闭 `a2a_strict_paper_architecture`，避免把实验扩展误称为论文
原始架构。

## 2. 两条计算路径

### 2.1 原有 MLP 路径

```text
history [B,H,A] --共享3层Conv1d+时间池化--> z0 [B,512]
future  [B,H,A] --同一个编码器-----------> z1 [B,512]

z_tau [B,512] --4-block AdaLN-MLP, condition=VLM+tau--> velocity [B,512]
z1_hat [B,512] --4-block residual MLP decoder----------> action [B,H,A]
```

### 2.2 新增 DiT 路径

```text
history [B,H,A] --共享3层Conv1d，不做时间池化--> z0 [B,H,C]
future  [B,H,A] --同一个编码器--------------> z1 [B,H,C]

z_tau [B,H,C]
  + relative-position embedding
  + real/padded-history validity embedding
  + VLM/tau AdaLN condition
        |
        v
N-block temporal DiT self-attention --> velocity [B,H,C]
        |
Euler integration
        v
z1_hat [B,H,C] --展平后4-block residual MLP decoder--> action [B,H,A]
```

其中：

- `B` 是 batch；
- `H` 默认是 8；
- `A` 是 padding 后动作维度；
- `C=a2a_dit_token_dim`，大模型默认配置为 256；
- 第 `i` 个 token 对应轨迹窗口中的第 `i` 个相对时间位置；
- 历史 `[-7,...,0]` 与未来 `[0,...,7]` 分别保持自身时间顺序，Flow 学习两段
  token 序列之间的整体传输，不假设第 `i` 对动作在物理上相等。

## 3. 为什么不是把 512 维向量随便切成 8 份

新编码器在最后一层 Conv1d 后保留真实时间轴，每个 token 都来自对应时间位置及其
局部卷积感受野。它不是先得到一个 512 维全局向量再 reshape。这样 Attention 的
token 才有明确的轨迹时序含义。

默认 DiT 配置 `H=8, C=256`，因此 token latent 总元素数是 2048；它是有意增大的
模型容量对照。若要做“潜变量总元素数相同”的结构消融，可设置：

```text
a2a_dit_token_dim = 64
8 tokens x 64 = 512
```

因此建议同时报告两种 DiT：

1. `DiT-512-budget`：8×64，用于尽量公平地比较 MLP 与 Attention；
2. `DiT-large`：8×256、8层，用于评估扩大容量后的上限。

## 4. DiT block 具体做什么

每个 `AdaLNDiTBlock` 包含：

1. 无 affine 的 LayerNorm；
2. 多头时间自注意力；
3. 残差连接；
4. 第二个无 affine LayerNorm；
5. GELU feed-forward MLP；
6. 第二个残差连接。

连续 Flow 时间 `tau` 先做正弦位置编码，再和 GR00T VLM 条件向量拼接。条件网络为
每个 block 产生两组 `shift / scale / gate`，分别调制 Attention 和 MLP：

```text
AdaLN(x) = LayerNorm(x) * (1 + scale) + shift
x_attn   = x + gate_attn * Attention(AdaLN_attn(x))
x_out    = x_attn + gate_mlp * MLP(AdaLN_mlp(x_attn))
```

这里的 Attention 只在动作 latent token 的时间轴上进行；VLM 条件通过 AdaLN 注入。
它没有改动 GR00T VLM 主干，也没有把视觉 token 复制进动作序列。

代码没有直接调用 `nn.MultiheadAttention`，而是显式实现 `QKV Linear -> MatMul ->
Softmax -> MatMul -> output Linear`。原因是当前仓库使用的传统 ONNX exporter 会把
`nn.MultiheadAttention` 融合成不受支持的 `_native_multi_head_attention`。显式实现
在数学上等价，并已经通过 PyTorch/ONNX Runtime 数值一致性与动态 batch/视觉序列
长度测试，也更容易被 TensorRT 识别为标准算子。

## 5. Flow Matching 没有改变

两种后端使用完全相同的 A2A 数学目标。共享编码器得到：

```text
z0 = E(history)
z1 = E(future)
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * z0 + tau * z1
target_velocity = z1 - z0
```

DiT 只是把张量从 `[B,D]` 改为 `[B,H,C]`。`tau` 会广播为 `[B,1,1]`，FM loss
仍然是预测速度与 `z1-z0` 的 MSE。AE、可微 Euler Integration Consistency、连续
动作 L1 和 auxiliary 动作损失也全部保留。

推理仍然从实际历史开始，而不是从高斯噪声开始：

```text
z = E(history)
for each Euler step:
    z = z + dt * DiT(z, tau, VLM_condition, history_validity)
future_action = Decoder(z)
```

## 6. 冷启动 token 的处理

刚开始不足 8 个真实历史点时，processor/policy 仍按既有规则产生数值补齐和
`history_action_mask`。Token encoder 会在输入适配前清零无效数值，并在每一层
时间卷积后再次清零无效位置，防止补齐值通过卷积泄漏。

DiT 不能简单删除这些位置，因为 8 个未来动作位置仍都需要生成。因此实现使用：

- 位置嵌入：告诉模型这是第几个相对时间 token；
- 有效性嵌入：告诉模型该历史 token 是真实反馈还是 cold-start padding；
- 不使用会删除 query 的 padding 逻辑：无历史的位置仍可从 VLM 条件和真实 token
  生成未来表示。

整个 episode 开始前仍必须调用 `policy.reset()`，且预测动作不能回填为 executed
history。

## 7. 配置字段

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `a2a_flow_backbone` | `mlp` | `mlp` 或 `dit` |
| `a2a_dit_token_dim` | 256 | 每个时间 token 的宽度 |
| `a2a_dit_num_layers` | 8 | DiT block 数量 |
| `a2a_dit_num_heads` | 8 | 每层注意力头数 |
| `a2a_dit_mlp_ratio` | 4 | DiT FFN 扩展比例 |
| `a2a_dit_dropout` | 0.0 | Attention/FFN dropout |
| `a2a_strict_paper_architecture` | true | 只允许论文核心 MLP；DiT 必须关闭 |

约束：

- `token_dim` 必须能整除 `num_heads`；
- 所有维度、层数、head 数和 ratio 必须大于 0；
- dropout 必须在 `[0,1)`；
- DiT 要求 history/future horizon 相等；
- 完整 A2A resume 时，backend、token width、层数、head 数等必须与 checkpoint 一致。

## 8. 训练命令

数据 canonicalization、channel specs 和 shared statistics 与原 MLP 完全共用。先按照
`getting_started/latent_a2a.md` 完成数据准备。

### 8.1 原 MLP 基准组

原命令无需增加任何字段，默认就是 MLP：

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <same_original_n1d7_checkpoint> \
  --dataset-path <preprocessed_dataset> \
  --embodiment-tag <tag> \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path <channel_specs.json> \
  --a2a-canonical-statistics-path <a2a_statistics.json> \
  --a2a-expected-contract-sha256 <contract_sha256> \
  --a2a-flow-backbone mlp \
  --a2a-training-stage joint \
  --global-batch-size 32 \
  --output-dir outputs/a2a_mlp
```

### 8.2 DiT-large 对照组

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <same_original_n1d7_checkpoint> \
  --dataset-path <same_preprocessed_dataset> \
  --embodiment-tag <same_tag> \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path <same_channel_specs.json> \
  --a2a-canonical-statistics-path <same_a2a_statistics.json> \
  --a2a-expected-contract-sha256 <same_contract_sha256> \
  --a2a-flow-backbone dit \
  --a2a-dit-token-dim 256 \
  --a2a-dit-num-layers 8 \
  --a2a-dit-num-heads 8 \
  --a2a-dit-mlp-ratio 4 \
  --a2a-dit-dropout 0 \
  --no-a2a-strict-paper-architecture \
  --a2a-training-stage joint \
  --global-batch-size 32 \
  --output-dir outputs/a2a_dit_large
```

### 8.3 相同 latent 元素预算的 DiT

把上面的 `--a2a-dit-token-dim 256` 改成 64。此时 `8×64=512`，更适合判断性能
变化来自 Attention 结构还是仅来自更大的 latent 容量。

如果采用 AE warmup，MLP 与 DiT 必须分别 warmup，并在各自的后续 joint 命令中加载
对应 checkpoint；不要把 MLP AE checkpoint 用作 DiT AE。

## 9. 公平对比要求

以下项目必须保持一致：

- 原始 GR00T base checkpoint；
- 数据集划分、canonical stats 和 contract hash；
- VLM 冻结/解冻方式；
- optimizer、学习率、batch size、训练步数和图像增强；
- history noise、Euler 步数和随机种子集合；
- 在线 history buffer、控制频率和执行 action chunk 的方式。

建议至少报告 3 个 seed，并记录：

- 闭环任务成功率；
- AE reconstruction L1；
- FM、IC latent、IC action loss；
- 物理单位动作误差、轨迹抖动和动作饱和率；
- action-head 参数量、峰值显存和延迟；
- 1、2、4、6 个 Euler step 的成功率/延迟折中。

单元测试通过只能证明公式、shape、mask、梯度和接口正确，不能替代真实机器人或
仿真闭环效果。更大的 DiT 也不能修复错误的物理动作语义；raw LIBERO 的 absolute
proprio 与 OSC delta command 不在同一连续空间，仍受原 A2A strict contract 门禁。

## 10. 代码位置

- `gr00t/configs/model/gr00t_n1d7_a2a.py`：backend 与 DiT 配置、严格模式门禁；
- `gr00t/configs/finetune_config.py`：训练 CLI 字段；
- `gr00t/model/modules/a2a_latent.py`：token encoder/decoder、AdaLN DiT block、
  temporal DiT FlowNet、通用 Euler；
- `gr00t/model/gr00t_n1d7_a2a/gr00t_n1d7_a2a.py`：双后端构造、FM/IC 形状广播、
  冷启动 validity mask 和推理分派；
- `gr00t/model/gr00t_n1d7_a2a/setup.py`：checkpoint 架构一致性与恢复字段；
- `scripts/deployment/export_onnx_n1d7_a2a.py`：导出 metadata 绑定 backend/DiT 参数；
- `scripts/deployment/a2a_trt_model_forward.py`：部署时核对 checkpoint 与 engine 后端；
- `tests/gr00t/model/test_a2a_latent.py`：token/DiT 模块专项测试；
- `tests/gr00t/model/test_a2a_action_head.py`：MLP 不回归、DiT 训练/推理/冷启动测试。

## 11. 怎样判断该选哪个

如果目标是复现论文核心结构或先取得稳定基线，使用 `mlp`。如果数据规模足够、动作
跨时间关系复杂，并且可以接受额外显存与延迟，再比较 `dit`。最终选择应依据闭环成功率
与实际部署预算，而不是只看 validation FM loss 或模型参数量。

## 12. 已完成的验收范围

实现写回时完成了以下检查：

- token encoder/decoder 的 shape、mask 隔离、反向梯度；
- 时间 Attention 的跨 token 信息传播；
- DiT 的 time/VLM conditioning 和 cold-start validity embedding；
- `[B,H,C]` latent 的 FM、AE、可微 Euler IC 和最终 action 输出；
- 默认大配置 `H=8, C=256, layers=8, heads=8` 的真实 CPU 前向；
- MLP 默认配置仍输出 `[B,512]`，没有新增可序列化参数键；
- MLP/DiT 完整 checkpoint 交叉恢复会 fail-fast；
- config 反序列化、训练 CLI 字段和部署 metadata；
- DiT ONNX checker、PyTorch/ORT 数值 oracle、动态 batch 和动态 VLM sequence；
- 原始 GR00T N1.7 Action Head、Processor 和模型 forward 回归。

当前环境没有 CUDA/TensorRT，因此没有实际构建 DiT TensorRT engine，也没有真实
canonical 数据训练和闭环成功率。ONNX 图和 TensorRT metadata/路由已经接通，但部署
机器上仍应执行 engine build、数值 verify 和实际延迟测试后再决定生产使用。
