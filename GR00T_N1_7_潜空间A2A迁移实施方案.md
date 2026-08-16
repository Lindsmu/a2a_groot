# GR00T N1.7 迁移潜空间 A2A 的详细实施方案（已按代码核查）

## 0. 范围与结论

- 论文：`C:\Users\a\Desktop\a_to_a\action_to_action_flow_matching.pdf`
- 目标仓库：`C:\Users\a\Desktop\a_to_a\Isaac-GR00T-main\Isaac-GR00T-main`
- 目标：在 GR00T N1.7 的 VLA 主干上，复现论文的 **512 维潜空间 Action-to-Action Flow Matching**，而不是把现有高斯噪声简单替换成历史动作。
- 本文范围：实现设计、逐文件修改清单、训练/推理/部署流程、测试与验收方案；当前不修改目标仓库。

结论：**可行，建议作为一个新的 `Gr00tN1d7A2A` 模型变体实现，保留原始 `Gr00tN1d7` 作为基线和启动兜底。**

这不是只改两行初始化代码的工作。GR00T 当前在原始归一化动作空间中做“高斯噪声 → 动作”的 4 步流匹配；论文方法要求增加历史执行轨迹、共享动作轨迹自编码器、512 维潜变量 FlowNet、潜变量一致性损失以及有状态推理。可复用的是 VLM 主干、数据/训练框架、归一化基础设施和策略 API；需要新建的是 A2A 数据路径和动作生成头。

建议第一版严格控制范围：

1. 只做一个 embodiment（优先 LIBERO，或最终要部署的单一机器人）。
2. 只选能够由 proprio state 和 future action 一一对应的连续通道。
3. 历史长度和未来长度都先设为 8，复现论文设定。
4. 潜变量固定 512 维，FlowNet 固定 4 个 AdaLN-MLP block。
5. 先完成 PyTorch 训练和仿真闭环，再做 ONNX/TensorRT。
6. 原始 GR00T 动作头完全保留，用于基线、首帧冷启动和失败回退。

不要在第一版同时做“全部 embodiment + 40 步动作块 + 离散 motion token + TensorRT”。这些问题混在一起后，很难判断失败来自 A2A 方法还是动作语义/部署改造。

---

## 1. 迁移后的模型到底是什么

### 1.1 当前 GR00T N1.7

当前动作生成逻辑是：

```text
图像 + 语言 ──> Qwen3-VL/Cosmos 主干 ──> VLM token 特征
当前 state ──> state encoder ───────────┐
高斯噪声动作序列 ──> action encoder ────┼─> 16 层 DiT ─> 动作速度 ─> 4 步 Euler ─> 动作块
                                          └─ cross-attend VLM token
```

ODE 状态是 `[B, H, D]` 的原始归一化动作张量。训练目标是 `action - noise`；推理从 `torch.randn` 开始。现有 `action_encoder` 只是逐步特征映射，没有与之配对的轨迹重建解码器，因此它不是论文中的动作自编码器。

### 1.2 目标潜空间 A2A

```text
实际执行的 proprio 历史 q[t-7:t] ─> 统一动作语义/归一化 ─> E_a ─> z0 ∈ R^512
未来示范动作 u[t:t+7] ────────────> 统一动作语义/归一化 ─> E_a ─> z1 ∈ R^512（仅训练）

图像 + 语言 ─> 原 GR00T VLM 主干 ─> 条件池化器 ─> c ∈ R^512

zτ=(1-τ)z0+τz1 ─> 4×AdaLN-MLP FlowNet(zτ,τ,c) ─> 预测 z1-z0
z0 ─> 1/2/4/6 步 Euler ─> ẑ1 ─> D_a ─> 未来动作块
```

这仍然是“在 GR00T VLA 模型基础上做的模型”：视觉语言主干和输入处理仍来自 GR00T；但原 16 层 DiT 动作生成头由论文式潜空间 A2A 头替换。第一版不建议把原 DiT 硬改成潜空间 MLP，因为那样既不忠实于论文，也会使计算量和旧假设残留。

---

## 2. 最关键的数据与动作表示契约

潜空间 A2A 能否成功，首先取决于 `z0` 和 `z1` 是否描述同一种物理量。不能直接把“历史关节状态”和“未来相对动作”丢给共享编码器。

### 2.1 时间定义

对观测时刻 `t`，第一版固定：

- 历史执行轨迹：`q[t-7], ..., q[t]`，共 8 个实际 proprio 采样。
- 未来目标轨迹：`u[t], ..., u[t+7]`，共 8 个示范动作。
- 当前图像和语言：时刻 `t`。
- 禁止读取其他 episode 的历史；训练时前 7 个时刻默认剔除，不使用负索引回绕。
- 推理冷启动时可以重复首个 proprio 填满 8 帧，但必须单独报告冷启动阶段性能。

A2A 变体使用的 modality 时间配置应明确改为：

| modality | `delta_indices` | 含义 |
|---|---|---|
| video | `[0]` | 当前图像 |
| language | `[0]` | 当前任务指令 |
| state | `[-7,-6,-5,-4,-3,-2,-1,0]` | 实际 proprio 历史 |
| action | `[0,1,2,3,4,5,6,7]` | 未来示范动作 |

processor 从 8 帧 state 构造 `history_action_canonical`；如果还要保留当前 state 条件，只取最后一帧 `state[:, -1:]`，因此旧的 `state_history_length` 无需改成 8。

这里的历史必须优先来自**实际反馈状态**，不能用上一轮模型尚未执行完的预测块代替。后者是 GR00T RTC 的“上一预测动作”，语义与论文 A2A 的已执行历史不同。

### 2.2 建立 `A2AChannelSpec`

为每个 embodiment 增加显式通道表，不能仅靠维度猜测：

```python
@dataclass
class A2AChannelSpec:
    action_key: str
    source_state_key: str | None
    kind: Literal["continuous", "binary", "categorical", "unsupported"]
    canonical_format: Literal["joint_position", "xyz_rot6d", "scalar", "token"]
    dim: int
```

规则：

- 连续关节位置：历史取对应实际 joint state；未来取原始绝对 target action。
- EEF：历史和未来必须统一为相同的 `XYZ+rot6d` 或相同 pose 格式，不能一侧 rotvec、一侧 rot6d。
- 连续夹爪位置：可以进入 A2A；二值开合建议先走辅助二分类头。
- `control_mode`、`motion_token` 等离散通道不进入连续潜空间流；进入辅助分类头或第一版直接排除。
- 如果某个 action key 找不到对应 `source_state_key`，该通道不能宣称为“实际执行历史 A2A”。

### 2.3 统一 canonical trajectory space

训练 processor 在调用现有 `StateActionProcessor.apply()` 之前，构造：

- `history_action_canonical`: `[8, Dmax]`
- `future_action_canonical`: `[8, Dmax]`
- `history_action_mask`: `[8, Dmax]`
- `future_action_mask`: `[8, Dmax]`
- `discrete_target`（如有）

建议 canonical 空间使用绝对物理轨迹：关节为绝对关节目标，EEF 为绝对 pose，连续夹爪为绝对位置。原因是 GR00T 的 `StateActionProcessor` 可能按 `ActionConfig` 把未来绝对动作转成相对动作，而历史 proprio 本身通常是绝对状态；如果直接复用转换后的动作，两侧不再同语义。

归一化也必须共享。为每个 canonical channel 计算一套 `a2a_statistics.json`，历史状态和未来目标都使用同一组尺度。建议使用二者合并后的稳健分位数，或使用明确的机器人物理上下限。不得让历史使用 state stats、未来使用 action stats 后便假设二者可比较。

### 2.4 数据边界与采样

现有数据集只按照未来 action horizon 缩短 episode，有效起点仍是 0。若把 state 的 `delta_indices` 改成 `[-7,...,0]`，早期样本在未 padding 时可能被 pandas 负索引错误地取到 episode 尾部。

因此新增 A2A 数据集类，统一计算：

```text
valid_start = max(0, -所有 modality 的最小 delta index)
valid_end   = episode_length - 1 - max(所有 modality 的最大 delta index)
valid_steps = [valid_start, ..., valid_end]
```

训练默认只采有效窗口；只有推理冷启动允许显式 padding，并产生 `history_valid_mask`。这项测试必须覆盖“第一个 episode 的前几帧不能读到最后几帧”。

### 2.5 第一版数据约束

在正式训练前，数据审计脚本必须给出：

1. 每个 action key 是否存在对应 state key。
2. 两侧维度、单位、坐标系、rotation format 是否一致。
3. episode 长度是否足以提供 8+8 窗口。
4. NaN、常量通道、异常跳变比例。
5. 归一化后 `||history-future||` 与 `||Gaussian-future||` 的分布。
6. 训练/验证集的 canonical stats 是否发生明显漂移。

若历史到未来的平均距离并未显著小于高斯到未来的距离，A2A 的核心前提在该数据上不成立，应先修正表示或重新选择通道，而不是直接训练。

---

## 3. 模型模块设计

### 3.1 配置

新增独立配置 `Gr00tN1d7A2AConfig`，`model_type="Gr00tN1d7A2A"`。建议第一版关键值：

```yaml
a2a_history_horizon: 8
a2a_future_horizon: 8
a2a_latent_dim: 512
a2a_encoder_conv_layers: 3
a2a_encoder_kernel_size: 5
a2a_flow_blocks: 4
a2a_flow_mlp_ratio: 4
a2a_decoder_res_blocks: 4
a2a_time_sampling: uniform
a2a_history_noise_train_std: 0.02
a2a_history_noise_infer_std: 0.0
a2a_num_inference_steps: 1
a2a_ic_train_steps: 1
a2a_lambda_fm: 1.0
a2a_lambda_ae: 0.5
a2a_lambda_ic: 1.0
a2a_lambda_ic_action: 0.5
a2a_include_current_state_condition: false
a2a_discrete_mode: auxiliary_head
```

`0.02` 是论文式的小历史噪声初值，只作用于归一化后的连续有效通道；同时必须做 `0 / 0.02 / 0.1` 消融。训练噪声和推理采样噪声分开配置，默认推理为 0、结果确定；需要研究多模态时才显式打开推理噪声。GR00T 的 Beta 时间采样不直接继承；严格 A2A 首版使用 `τ~Uniform(0,1)`。

现有 `state_history_length` 保持 1。把它改成 8 只会扩大旧 state encoder 的输入，并不能产生历史动作 A2A，而且会造成基础 checkpoint 形状不匹配。

### 3.2 `ActionTrajectoryEncoder E_a`

输入：`trajectory [B,8,132]`、trajectory mask、embodiment id。

推荐结构：

1. 每个时间步先乘有效通道 mask。
2. 使用 `CategorySpecificLinear(132→128)` 做 embodiment-specific 输入适配。
3. 转为 `[B,C,T]`。
4. 3 层 Conv1d，kernel=5、padding=2，通道例如 `128→256→512→512`，每层 SiLU/GELU + normalization。
5. 按时间 mask 做 masked pooling，得到 `[B,512]`。

历史和未来必须共享同一个 `E_a`。第一版固定两侧长度都是 8，以免另加 source/target encoder 后偏离论文。

### 3.3 `LatentActionDecoder D_a`

输入 `[B,512]`，经过 4 个 residual MLP block，再用 embodiment-specific output projection 输出 `[B,8,132]`。输出立即乘 future mask。

第一版直接使用 residual MLP，贴近论文。以后若恢复 40 步 horizon，再评估 temporal decoder；不要在严格复现阶段提前替换结构。

### 3.4 VLM 条件池化器

GR00T 主干输出 `[B,S,2048]` token 序列，而论文 FlowNet 使用一个条件向量。增加 `A2AConditionPooler`：

1. 复用当前 `vlln`。
2. 利用 `backbone_attention_mask` 排除 padding。
3. 若有 `image_mask`，分别池化图像 token 和文本 token。
4. 拼接两个摘要并投影到 `[B,512]`。

可用 learned attentive pooling；若为了最小复现，先用 masked mean + MLP，并把池化方式作为消融。不能简单取序列最后一个 token，因为左 padding、图像 token 布局和不同语言长度会改变其语义。

严格论文模式不额外输入旧 state encoder。历史 proprio 已包含当前机器人状态。可增加 `include_current_state_condition` 消融，但默认关闭，避免把“论文效果”和“额外 state shortcut”混在一起。

### 3.5 512 维 FlowNet

FlowNet 输入 `zτ [B,512]`、连续时间 embedding 和 `c [B,512]`。使用 4 个 AdaLN residual MLP block：

```text
h = zτ
for block in 4 blocks:
    scale, shift, gate = MLP(time_embed(τ) + condition(c))
    h = h + gate * MLP(scale * LayerNorm(h) + shift)
v̂ = output_projection(h)
```

输出仍为 `[B,512]`，表示潜变量速度。此处不复用当前 16 层 DiT；现有 DiT 的序列 token、cross-attention、hidden size 和动作 decoder 都服务于原始动作空间流。

### 3.6 离散通道

论文方法主要适用于连续、平滑动作。对 GR00T 的离散通道采用明确的混合策略：

- A2A decoder 只输出连续通道。
- `AuxiliaryDiscreteHead(c, ẑ1)` 输出每个 future step 的 binary logits 或 categorical logits。
- binary 使用 BCEWithLogits，categorical 使用 CrossEntropy。
- 输出阶段按 `A2AChannelSpec` 合并回原 action dict。

第一版若选 LIBERO/单臂任务，可以先把夹爪按现有数据语义做连续标量，并另做二值化消融；`motion_token`、`control_mode` 不应被当作连续坐标流动。

---

## 4. 训练目标与前向流程

### 4.1 前向伪代码

```python
hist = inputs.history_action_canonical
future = inputs.future_action_canonical
hmask = inputs.history_action_mask
fmask = inputs.future_action_mask

if training and sigma > 0:
    hist = hist + sigma * randn_like(hist) * hmask

z0 = action_ae.encode(hist, hmask, embodiment_id)
z1 = action_ae.encode(future, fmask, embodiment_id)

c = condition_pooler(
    backbone_features,
    backbone_attention_mask,
    image_mask,
)

tau = uniform(0, 1)
z_tau = (1 - tau) * z0 + tau * z1
target_velocity = z1 - z0
pred_velocity = flow_net(z_tau, tau, c)

loss_fm = mse(pred_velocity, target_velocity)
future_ae = action_ae.decode(z1, embodiment_id) * fmask
loss_ae = masked_l1(future_ae, future, fmask)

z1_hat = differentiable_euler(flow_net, z0, c, steps=ic_steps)
future_hat = action_ae.decode(z1_hat, embodiment_id) * fmask
loss_ic_latent = l1(z1_hat, z1)
loss_ic_action = masked_l1(future_hat, future, fmask)

loss_ic = loss_ic_latent + 0.5 * loss_ic_action
loss = 1.0 * loss_fm + 0.5 * loss_ae + 1.0 * loss_ic + loss_discrete
```

所有 action-space loss 都必须使用 mask 后再按有效元素数量归一化，不能让 132 维 padding 或不足 8 步的 padding 进入损失。

### 4.2 三个核心损失

- `L_FM`：潜空间 flow matching，监督 `F(zτ,τ,c)` 逼近 `z1-z0`。
- `L_AE`：未来动作的重建 L1，防止 encoder/decoder 失去动作语义。
- `L_IC`：从历史潜变量实际积分得到的 `ẑ1`，同时在潜空间和解码动作空间接近未来目标。

最终权重使用论文值：`λFM=1, λAE=0.5, λIC=1, λIC-action=0.5`。工程上建议给 `λIC` 做短 warmup：先从 0 线性升到 1，避免未训练 FlowNet 的积分误差在最初阶段压倒其他损失；最终权重不变，并同时保留“不 warmup”的严格复现实验。

### 4.3 防止潜变量坍缩

必须持续记录：

- 每维 latent mean/std。
- `||z0-z1||` 的均值和分位数。
- decoder 重建误差（归一化与物理单位）。
- `L_FM / L_AE / L_IC-latent / L_IC-action`。
- 有效动作范围、输出饱和比例。

若 `z0,z1` 的标准差趋近 0，而 AE 误差没有同步下降，就是 latent collapse。第一版不应立刻增加复杂正则；先检查 mask、decoder 输出和损失归一化是否正确，再考虑 latent variance regularization。

### 4.4 推荐训练阶段

**阶段 A：数据与表示验证**

- 跑完整 canonical mapping/统计审计。
- 随机抽样可视化历史与未来轨迹。
- 验证 absolute↔relative 和 EEF format round-trip。

**阶段 B：AE warmup**

- 冻结 VLM 主干。
- 只训练 `E_a/D_a`，目标为未来轨迹重建。
- 可增加一个低权重历史重建作为工程消融，但严格论文结果仍使用论文定义的 future AE loss。
- 直到验证集物理误差进入控制器可接受范围，再进入 flow 训练。

**阶段 C：联合 A2A**

- VLM 继续冻结；训练 condition pooler、AE、FlowNet 和辅助离散头。
- 使用 `L_FM + L_AE + L_IC`。
- 有效 batch size 对齐论文的 32；显存不足时用 gradient accumulation。
- 历史连续通道加入 mask 后的小噪声。

**阶段 D：一步推理优化**

- 以 1 步为主要目标，同时验证 2/4/6 步。
- 若 1 步明显落后而 4 步正常，优先检查 IC 训练积分和条件池化，不要先扩大模型。
- 可在最后少量解冻 condition projection 或 VLM 顶层；必须与全冻结结果分开报告。

**阶段 E：任务微调**

- 在目标机器人/任务上微调。
- 保留同数据量、同 backbone、同训练步数的原 GR00T 基线。

优化器、学习率和 warmup 可以先继承项目现有 finetune 配置；新 head 的初始学习率可从 `1e-4` 量级开始小范围搜索，VLM 若解冻则使用至少低一个数量级的学习率。这个范围是工程初值，不是论文声称的固定超参数。

---

## 5. Checkpoint 迁移策略

不能直接让现有 `AutoModel.from_pretrained()` 读取新结构。仓库的 setup 会收集 missing/unexpected/mismatched keys，并对除 mask token 外的任何差异抛错。

新增显式迁移入口，例如：

```text
--model-config Gr00tN1d7A2A
--training.base-vla-checkpoint <原 GR00T checkpoint>
--training.start-from-checkpoint <仅用于恢复同结构 A2A checkpoint>
```

两种加载语义必须分开：

1. **base-vla-checkpoint**：只迁移允许列表中的旧权重。
   - `backbone.*`
   - 与新 condition path 形状一致时的 `action_head.vlln.*`
   - 可选 `vl_self_attention.*`，但需明确是否在新模型使用。
2. **start-from-checkpoint**：恢复完整 A2A 模型、优化器和 trainer state，要求严格同构。

实现时打印并保存：已复用键、A2A 新初始化键、被忽略的旧动作头键和任何非法形状差异。不要全局使用 `ignore_mismatched_sizes=True`，否则 backbone 误加载也可能被吞掉。

新 checkpoint 至少包含：

- `config.json`：唯一 model type 和全部 A2A 配置。
- `processor_config.json`：history/future horizon、channel spec、cold-start 策略。
- `a2a_statistics.json`：canonical 归一化统计。
- `embodiment_id.json`。
- 权重与训练状态。

---

## 6. 推理与策略集成

### 6.1 推荐的输入责任边界

最可靠方式是由机器人客户端随每次请求发送最近 8 帧实际 proprio，服务器不自行猜测。原因：丢包、执行延迟、动作裁剪、急停和控制器跟踪误差都会让服务器保存的“预测历史”偏离真实执行历史。

优先复用现有 observation 协议：A2A config 的 state temporal length 是 8，所以 `observation["state"][key]` 直接携带 `[B,8,D]` 历史，不必再发明一个新的顶层 modality。processor 将最后一帧视为 current state，将完整 8 帧映射为 A2A history。

输入示意：

```python
observation = {
    "video": ...,
    "state": {key: actual_q_t_minus_7_to_t},  # [B, 8, D]
    "language": ...,
}
```

为了兼容旧客户端，可以在 `Gr00tA2APolicy` 中提供 ring buffer，但要有：

- `reset()`：新 episode、急停、任务切换时清空。
- 时间戳/控制周期校验。
- 丢帧处理和最大允许间隔。
- 明确标记 history 来源是 actual feedback 还是 fallback。

### 6.2 冷启动

建议提供三种模式并在评估时分别记录：

1. `repeat_first_state`：重复首个 proprio 8 次；实现最简单。
2. `initial_history_file`：复用项目 initial action/state 机制保存标准起始历史。
3. `legacy_fallback`：前 7 个执行步使用原 GR00T 动作头，缓冲区满后切换 A2A；最稳健但需要同时加载旧动作头。

第一版训练/仿真可用 `repeat_first_state`，真实机器人部署建议 `legacy_fallback` 或明确的启动轨迹。

### 6.3 A2A 推理

```python
z = E_a(history_action_canonical)
for k in range(K):
    tau = k / K
    z = z + (1 / K) * FlowNet(z, tau, condition)
continuous_future = D_a(z)
discrete_future = AuxiliaryDiscreteHead(condition, z)
action_dict = decode_and_merge(continuous_future, discrete_future)
```

输出 canonical 绝对动作后直接反归一化到物理动作格式；不要再次套用“相对动作→绝对动作”的旧解码路径。若目标控制接口仍需要 relative action，则在最后明确做一次 canonical absolute→controller representation 转换。

### 6.4 与 RTC 的关系

A2A 解决的是生成分布起点和少步推理；RTC 解决的是相邻预测块重叠和异步延迟。两者可以叠加，但第一阶段应分开验证。

现有 RTC 使用上一轮**未执行完的预测动作**做 inpainting，并且仓库文档明确说明它尚未接入 `Gr00tPolicy`/server-client。不能把这条路径当作论文历史执行轨迹的现成实现。

---

## 7. 逐文件实施清单

以下以新增变体为主，避免破坏原模型。

### 7.1 新增文件

| 文件 | 主要职责 |
|---|---|
| `gr00t/configs/model/gr00t_n1d7_a2a.py` | A2A config、唯一 model type、注册配置 |
| `gr00t/model/modules/a2a_latent.py` | trajectory encoder/decoder、condition pooler、AdaLN FlowNet、Euler solver |
| `gr00t/model/gr00t_n1d7_a2a/gr00t_n1d7_a2a.py` | 新模型和 action head；复用原 VLM backbone |
| `gr00t/model/gr00t_n1d7_a2a/processing_gr00t_n1d7_a2a.py` | canonical history/future、mask、decode/merge、processor 保存加载 |
| `gr00t/model/gr00t_n1d7_a2a/setup.py` | 新 pipeline、数据选择、base checkpoint 迁移器 |
| `gr00t/data/dataset/a2a_single_step_dataset.py` | 支持负历史窗口的有效索引和无泄漏抽样 |
| `gr00t/policy/gr00t_a2a_policy.py` | proprio history/ring buffer、reset、cold-start、A2A decode |
| `scripts/a2a/audit_a2a_data.py` | 通道映射、单位、距离、异常和窗口审计 |
| `scripts/a2a/build_canonical_stats.py` | 生成共享 canonical stats |
| `scripts/deployment/export_onnx_n1d7_a2a.py` | A2A 专用 ONNX 导出 |
| `scripts/deployment/a2a_trt_model_forward.py` | 从历史 latent 开始的 TRT Euler loop |

### 7.2 修改现有文件

| 文件 | 修改点 |
|---|---|
| `gr00t/configs/model/__init__.py` 或配置导入入口 | 注册 `Gr00tN1d7A2AConfig` |
| `gr00t/data/types.py` | 可选增加显式 A2A history 字段；若 processor 直接从多帧 states 构造则可不改 dataclass |
| `gr00t/data/dataset/factory.py` | 按 A2A config 选择 A2A dataset 类 |
| 模型包导入入口 | 触发 AutoConfig/AutoModel/AutoProcessor 注册 |
| `gr00t/experiment/trainer.py` | 可选：记录多项 A2A loss；总 loss 训练本身可继续走 HF Trainer |
| policy/server 输入协议 | 增加实际 proprio history；保留旧协议兼容模式 |

### 7.3 不建议直接修改的核心路径

- 不要把 `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` 原动作头原地改成 A2A。
- 不要把 `state_history_length` 改成 8 来冒充 action history。
- 不要把现有 `action_encoder` 当成论文 `E_a`；它没有轨迹级 512 维瓶颈和重建约束。
- 不要让原 `trt_model_forward.py` 继续从 `torch.randn` 初始化；A2A TRT 必须从历史 latent 初始化。

---

## 8. 测试计划

### 8.1 数据与 processor 单测

1. episode 起点没有负索引回绕。
2. episode 末尾没有 future 越界。
3. `history=[t-7..t]`、`future=[t..t+7]` 无 off-by-one。
4. canonical absolute↔controller representation round-trip。
5. EEF rotation format round-trip。
6. history 与 future 使用同一 stats。
7. padding 的值不影响 encoder 输出和 loss。
8. 离散通道不会进入连续 A2A mask。

### 8.2 模块单测

1. `E_a: [B,8,132]→[B,512]`。
2. `D_a: [B,512]→[B,8,132]`。
3. FlowNet 输出 shape、dtype 和梯度正确。
4. `τ=0/1` 的插值端点正确。
5. 用 oracle constant velocity 时，Euler 能得到已知终点。
6. IC 路径不在训练时误用 `torch.no_grad()`，梯度能到 FlowNet 和 decoder。
7. mask 之外的随机值改变时，输出/损失保持不变。
8. `sigma=0` 时相同输入推理确定；`sigma>0` 时只扰动有效连续历史。
9. AE 能在小数据上过拟合并显著降低重建误差。
10. latent std 不为 0，decoder 输出无 NaN/Inf。

### 8.3 集成测试

1. 10～100 条轨迹 tiny-overfit：三项 loss 均可下降。
2. 原 `Gr00tN1d7` 全部测试继续通过。
3. 新模型从 base checkpoint 只加载允许列表，非法 backbone mismatch 必须失败。
4. 完整 A2A checkpoint save/load 后输出一致。
5. policy history buffer 在 `reset()` 后清空。
6. 客户端实际 history 与服务器 ring buffer 两种输入得到一致结果。
7. 冷启动到缓冲区满的切换不产生动作突跳。
8. PyTorch 与 ONNX/TRT 在固定输入上的误差满足设定阈值。

### 8.4 闭环评估矩阵

至少比较：

| 组别 | 起点空间 | 潜空间 | 推理步数 |
|---|---:|---:|---:|
| 原 GR00T | Gaussian→raw action | 否 | 4 |
| 原 GR00T | Gaussian→raw action | 否 | 1 |
| Raw A2A 对照 | history→raw action | 否 | 1/4 |
| Latent A2A，无 IC | history→512 latent | 是 | 1/4 |
| Latent A2A，完整 | history→512 latent | 是 | 1/2/4/6 |

每组保持相同 backbone、数据拆分、训练步数和随机种子，至少 3 个 seed。报告：

- 任务成功率和分任务成功率。
- 视觉/语言 OOD 成功率。
- 初始姿态扰动、历史噪声、丢帧下的鲁棒性。
- action MAE（物理单位）、chunk 边界跳变、速度/加速度/jerk。
- action-head latency、VLM latency、处理 latency 和端到端 latency。
- 参数量、显存和吞吐量。
- 离散夹爪/模式准确率。

### 8.5 推荐的 go/no-go 门槛

这些是工程验收初值，不是论文原始数字：

1. **数据门**：canonical history→future 距离明显小于 Gaussian→future；否则先停止模型训练。
2. **AE 门**：物理单位重建误差低于目标控制器的可接受死区/跟踪误差。
3. **模型门**：latent 无坍缩，1/2/4 步随步数增加表现合理，不出现“4 步反而系统性变差”。
4. **任务门**：1 步 A2A 成功率建议不低于原 4 步 GR00T 5 个百分点以上；若业务要求更高，应按业务门槛收紧。
5. **速度门**：必须测端到端，而不是只把 4 步变 1 步就宣称 4 倍加速。

仓库现有 H100 TensorRT 数据中，4 步模型约为：数据处理 6.2 ms、backbone 8.8 ms、action head 12.3 ms、E2E 27.9 ms。即使动作头理想降到接近 0，端到端理论上限也约是 `27.9/(6.2+8.8)≈1.86×`，实际会更低。Orin 更受 backbone 限制。因此 A2A 的价值应同时看成功率、平滑性和 action-head 延迟，不能只看 ODE 步数。

---

## 9. 分阶段实施顺序与交付物

### M0：冻结实验协议

- 确认第一个 embodiment、数据集、控制频率和 8 步对应的实际时间长度。
- 完成 `A2AChannelSpec`。
- 确认 continuous/discrete 划分和 cold-start 策略。

验收：一份机器可读 channel spec，所有通道的单位、坐标系和来源明确。

### M1：数据窗口与审计

- 实现 A2A dataset、canonical processor 和 stats 脚本。
- 跑边界、round-trip、距离分布测试。

验收：无 episode 泄漏；随机样本人工检查正确；A2A 源分布确实比 Gaussian 更接近目标。

### M2：动作 AE

- 实现 3 层 Conv1d encoder、512 维瓶颈、4 层 residual decoder。
- 完成 tiny-overfit 与验证集重建。

验收：物理误差满足控制要求，latent 无坍缩。

### M3：潜空间 flow 与损失

- 实现 condition pooler、4-block AdaLN FlowNet、可微 Euler、FM/AE/IC loss。
- 冻结 VLM 进行联合训练。

验收：三项 loss 正常下降；1/2/4/6 步离线预测和小规模闭环可运行。

### M4：Checkpoint 与训练流水线

- 实现 base VLA 权重允许列表加载。
- 实现 A2A 完整恢复、保存 processor/stats/config。
- 增加多项 loss/latent 指标日志。

验收：基础 checkpoint 迁移报告清晰；A2A checkpoint 严格恢复一致。

### M5：Policy 与闭环

- 接入 actual proprio history、reset、cold-start、动作合并。
- 完成仿真基线和消融矩阵。

验收：同预算下完成至少 3 seeds；达到任务与稳定性门槛。

### M6：部署优化

- 固定 H=8/Dmax 后导出 A2A 模块。
- 验证 ONNX/TRT 数值；更新 benchmark。

验收：PyTorch/部署后端闭环差异可接受，报告分模块和端到端延迟。

---

## 10. 已完成的代码自查

| 核查项 | 仓库现状 | 对迁移的结论 |
|---|---|---|
| 当前 flow 起点 | `gr00t_n1d7.py` 约 230～278 行：训练从 `randn` 构造 raw action path | 必须新建 latent flow，不能只复用 loss |
| 当前推理 ODE | 同文件约 349～435 行：raw action tensor 从 `randn` 做 Euler | A2A 要改为从 `E(history)` 的 512 latent 开始 |
| 当前动作编码器 | `MultiEmbodimentActionEncoder` 只产生 DiT token embedding | 不是轨迹 AE；需新 `E_a/D_a` |
| state history | config 默认 `state_history_length=1` | 改为 8 不是 A2A，且会破坏旧权重形状 |
| 数据结构 | `VLAStepData` 只有 states/actions，没有 action history | 可由多帧 states 在 A2A processor 显式构造 |
| 数据允许模态 | loader 仅允许 video/state/action/language/mask | 不要新增未知 `action_history` modality；用 state 负窗口或新 dataset 路径 |
| episode 边界 | 当前有效长度只减 future action horizon，step 从 0 开始 | 必须增加 valid_start，防止负 iloc 回绕 |
| processor 顺序 | raw action 进入 `StateActionProcessor.apply()` 后才转 relative/normalize | canonical future 应在此调用之前构造 |
| collator | 非 VLM 字段统一 `np.stack` | 新 history/mask 张量可直接批处理 |
| 多动作语义 | `ActionConfig` 支持 relative/delta/absolute、EEF/non-EEF、不同 format | 必须建立 channel spec 和统一 canonical 空间 |
| 严格权重加载 | setup 对 missing/unexpected/mismatched key 抛错 | 要新增 base-checkpoint 迁移器，不能直接 from_pretrained |
| policy history | policy 目前只处理当前 observation，且调用模型时没有传 options | 必须增加 actual proprio history 和 reset 语义 |
| RTC | 只在低层 action head；使用上一预测块，未接 policy/server | 不能替代论文 A2A 的已执行历史 |
| 导出部署 | ONNX/TRT 脚本硬编码 state/action encoder、DiT、decoder 和 `randn` loop | 应新增 A2A 专用导出和 forward，不覆盖原脚本 |
| trainer | HF Trainer 读取模型返回的总 `loss` | 总训练流程可复用；额外损失日志需补充 |
| 动作连续性 | 部分 embodiment 含 motion token/control mode/discrete gripper | 连续 A2A + 辅助离散头，或第一版排除这些通道 |

自查后的最终判断：**方法迁移在架构上没有硬性障碍，真正的高风险点不是 512 维 FlowNet，而是历史 proprio 与未来动作的物理语义对齐、episode 边界、离散通道和推理时真实历史来源。**只要这四项按本方案先做验证，GR00T 的 VLM 主干和训练框架可以稳定复用。

---

## 11. 开始编码前必须锁定的 5 个项目参数

1. 第一个复现对象：LIBERO、SO100、G1，还是自有机器人数据。
2. 真实控制频率；8 帧历史和 8 帧未来分别覆盖多少毫秒。
3. 每个 action key 对应哪个实际 state key，以及单位/坐标系。
4. 哪些通道连续、哪些离散、哪些暂不支持。
5. 冷启动使用重复首状态、标准初始轨迹还是旧 GR00T fallback。

若尚未指定，建议默认从 **LIBERO 单 embodiment、Hhist=Hfuture=8、连续动作 + 单独夹爪消融、repeat-first-state 冷启动、冻结 VLM** 开始。这条路线最容易先验证论文结论，再扩展到真实机器人。
