# GR00T N1.7 潜空间 A2A：完整实现与使用手册

本文是仓库中 `Gr00tN1d7A2A` 默认 MLP 后端的实现参考与操作手册，覆盖“实现了什么、每个模块如何工作、数据怎样准备、怎样训练/恢复/推理/部署、怎样验收，以及当前不能做什么”。快速操作可先看 `getting_started/latent_a2a.md`；需要理解或修改默认论文核心迁移时阅读本文。新增的时间序列 latent-token DiT 是并列实验后端，不改变本文所述默认行为，单独见 [`潜空间A2A_DiT对比实现与使用.md`](潜空间A2A_DiT对比实现与使用.md)。

## 1. 最终实现结论

这是一个建立在 GR00T N1.7 VLA 基础上的新模型变体，不是完全独立的新视觉语言模型，也不是在旧 DiT 中把 `randn` 简单换成历史动作。

- 保留：GR00T N1.7 的视觉语言主干、token 化、数据框架、训练框架、policy 接口和多 embodiment 基础设施。
- 替换：原始“高斯噪声 → 原始动作空间”的 DiT action head。
- 新增：历史 executed-proprio 窗口、共享轨迹自编码器、默认 512 维潜空间和 4-block AdaLN-MLP FlowNet、可选时间序列 DiT FlowNet、AE/IC 损失、有状态历史 buffer、canonical 数据契约和 A2A 专用 ONNX/TensorRT 路径。
- 不影响：原始模型类型仍为 `Gr00tN1d7`；A2A 使用独立 `model_type="Gr00tN1d7A2A"`，不会覆盖旧模型注册。

实现的论文核心是：把最近执行历史编码成 `z0`，把未来动作编码成 `z1`，在共享 512 维潜空间学习从 `z0` 到 `z1` 的条件速度场。推理从 `z0` 开始，因此不再创建高斯动作初值。

## 2. 论文方法与本项目实现的边界

### 2.1 与论文一致的核心

给定执行历史 `a_hist`、未来示范 `a_future` 和视觉语言条件 `c`：

```text
z0 = E_a(a_hist)
z1 = E_a(a_future)              # 只在训练时需要
tau ~ Uniform(0, 1)
z_tau = (1-tau) z0 + tau z1
v_target = z1 - z0
v_pred = f_theta(z_tau, tau, c)
```

同一个 `ActionTrajectoryEncoder` 同时计算 `z0` 和 `z1`。三个论文损失被逐项实现：

```text
L_FM = MSE(v_pred, z1-z0)
L_AE = L1(D_a(E_a(a_future)), a_future)
z1_hat = Euler(f_theta, z0, c)
L_IC = L1(z1_hat, z1) + 0.5 * L1(D_a(z1_hat), a_future)
L_total = 1.0*L_FM + 0.5*L_AE + 1.0*L_IC
```

严格连续动作时，上式就是实际总损失。默认 horizon 是历史 8 步、未来 8 步；latent 是 512 维；encoder 是 3 层 kernel-5 Conv1d；FlowNet 是 4 个 AdaLN-MLP residual block；decoder 是 4 个 residual MLP block。推理默认做 1 步显式 Euler，也支持 PyTorch 路径的多步对照。

### 2.2 为 GR00T 增加的工程适配

下列部分不是论文原网络的逐比特复刻：

- 论文的 ResNet18 + linear condition 被 GR00T VLM + 语言 token + `VLMConditionPooler` 替代。
- 论文实验写 `m=8` 帧视觉历史；迁移方案保留 GR00T 当前/原配置视觉 horizon，示例是单帧 `[0]`。
- 多 embodiment 输入/输出由 `CategorySpecificLinear/MLP` 适配。
- binary、categorical、不能与 proprio 可逆对应的 regression 通道进入 auxiliary head，不污染连续 A2A latent。
- canonical statistics、版本化控制器契约、cold start、checkpoint 迁移、ONNX/TensorRT 都是生产化补强。

论文没有公开 encoder 中间通道宽度、MLP ratio、激活、归一化、time embedding 和 IC 训练积分步数。本仓库固定 `(128,256,512)`、ratio 4、SiLU/LayerNorm、FP32 sinusoidal time embedding 和 1 步 IC Euler；这些是可复现的仓库实现选择，不能声称是未公开作者代码的精确数值。

### 2.3 尚未复现的论文效果

单元测试只能证明算法和软件接口正确，不能等价为成功率复现。当前仓库没有随实现一起提供 canonical 机器人数据、训练完成的 A2A checkpoint、3-seed 闭环结果或真实端到端延迟。因此准确表述是：核心方法实现正确，真实任务效果待数据转换、训练和闭环实验验证。

## 3. 完整张量流

默认 `B` 为 batch，`H=F=8`，`D=max_action_dim`，`L=512`：

```text
history_action_canonical [B,H,D] ─┐
history_action_mask      [B,H,D] ─┴─> shared E_a ─> z0 [B,L]

future_action_canonical  [B,F,D] ─┐
continuous_action_mask   [B,F,D] ─┴─> same E_a ─> z1 [B,L] (train only)

backbone_features [B,S,C] + attention/image masks
    └─> masked global/image/text pooling ─> c [B,L]

z_tau, tau, c ─> 4×AdaLN-MLP ─> latent velocity [B,L]
z0 ─> differentiable Euler ─> z1_hat [B,L]
z1_hat ─> 4×residual MLP decoder ─> continuous future [B,F,D]
c + z1_hat ─> auxiliary head ─> regression/binary/categorical future
mask merge ─> action_pred [B,F,D]
```

mask 既决定哪些值进入 encoder/loss，也决定输出位置。模型逐样本检查至少存在一个 continuous 通道；history 和 future continuous support 必须完全一致；重叠 head mask、未覆盖有效位、padding 位上错误生成 mask、NaN/Inf 和错误 horizon 都会立即失败。

## 4. 文件级实现索引

### 4.1 配置与注册

- `gr00t/configs/model/gr00t_n1d7_a2a.py`：A2A 模型配置、论文核心 profile 和 loss/noise/solver/stage/contract 参数。
- `gr00t/configs/base_config.py`：从 YAML/dict 恢复时根据 `model_type` 实例化正确子类。
- `gr00t/model/__init__.py`、`gr00t/model/gr00t_n1d7_a2a/__init__.py`：Transformers `AutoConfig/AutoModel/AutoProcessor` 与项目 pipeline 注册。
- `gr00t/configs/finetune_config.py`：标准 finetune CLI 的 A2A 入口参数。

### 4.2 数据窗口与契约

- `gr00t/data/dataset/a2a_single_step_dataset.py`：对所有 modality delta 求严格交集，禁止负索引回绕、episode 越界和训练 padding。
- `gr00t/data/dataset/factory.py`：A2A model 自动选择严格 dataset。
- `gr00t/data/types.py`：`A2AChannelSpec`，声明 continuous/regression/binary/categorical/unsupported 和语义元数据。
- `gr00t/data/a2a_contract.py`：验证 version、N/H/F/D、通道顺序、控制器 provenance 并计算稳定 SHA-256。
- `scripts/a2a/export_canonical_windows.py`：从已经 canonicalized 的 LeRobot 列导出未归一化物理窗口 NPZ。
- `scripts/a2a/audit_a2a_data.py`：强制验证 exporter metadata、双 mask、contract hash、finite、跳变、常量通道和 history 距离优势。
- `scripts/a2a/build_canonical_stats.py`：只对 history/future mask 的 continuous 交集计算一套共享统计并绑定 contract/statistics SHA。

### 4.3 模型与损失

- `gr00t/model/modules/a2a_latent.py`：trajectory encoder/decoder、VLM pooler、AdaLN FlowNet、masked L1、可微 Euler。
- `gr00t/model/gr00t_n1d7_a2a/gr00t_n1d7_a2a.py`：FM/AE/IC/aux loss、分阶段冻结、训练诊断、推理和完整 VLM+A2A 模型。
- `gr00t/model/gr00t_n1d7_a2a/processing_gr00t_n1d7_a2a.py`：从 state/action 构造 canonical 输入、共享归一化、严格语义/统计门禁和动作解码。
- `gr00t/model/gr00t_n1d7_a2a/setup.py`：horizon 配置、base N1.7 allowlist 迁移、完整 A2A resume、processor 构建和阶段覆盖。

### 4.4 训练与推理状态

- `gr00t/experiment/launch_finetune.py`：标准 CLI 路由到 A2A pipeline；自动关闭旧 relative-action transform。
- `gr00t/experiment/trainer.py`：每个真实 optimizer step 记录 FM/AE/IC/aux、latent std 和 pair distance。
- `gr00t/policy/a2a_history.py`：只保存真实 proprio 的环形 buffer，验证 shape、时间戳单调性、最大间隔、reset 和 cold-start mask。
- `gr00t/policy/gr00t_policy.py`：支持外部完整 `[B,8,D]` history 或逐帧积累；从不把预测 action 写入 executed history。

### 4.5 部署

- `scripts/deployment/export_onnx_n1d7_a2a.py`：导出 10 输入、固定 solver steps 的 fused FP32 action head；默认 ONNX checker + PyTorch/ORT 数值 oracle。
- `gr00t/deployment/a2a_artifacts.py`：checkpoint、statistics、ONNX、engine 和 canonical JSON 的稳定摘要。
- `scripts/deployment/build_tensorrt_engine.py`：`a2a_action_head` 专用 FP32 构建/profile，构建前校验 ONNX 和所有 identity。
- `scripts/deployment/a2a_trt_model_forward.py`：按 engine dtype 转换、设置动态 shape、运行时绑定 checkpoint/统计/数据 contract/engine。
- `gr00t/deployment/modes.py`、`scripts/deployment/trt_model_forward.py`、`benchmark_inference.py`：标准 build/setup/benchmark/verify 路由；A2A 数据用严格有效 anchor，不从 episode 尾部负索引历史。

## 5. 数据必须先满足什么

continuous A2A 的历史和未来必须是相同的物理轨迹定义：相同格式、单位、坐标系、旋转复合、控制器版本、时序和 target 定义。相同维度或相同列名不是证据。

每个 continuous channel 至少声明：

```json
{
  "action_key": "eef_pose_canonical",
  "source_state_key": "eef_pose_canonical",
  "kind": "continuous",
  "dim": 9,
  "canonical_format": "xyz_rot6d",
  "source_format": "xyz_rot6d",
  "target_format": "xyz_rot6d",
  "source_unit": "meter_unitless",
  "target_unit": "meter_unitless",
  "source_frame": "world",
  "target_frame": "world"
}
```

contract 还必须绑定 dataset immutable revision/fingerprint、原 schema SHA、canonicalizer name/version/code SHA、controller type/version/scale/control_delta/frame/rotation composition、target definition 和 observation/action time alignment。

### 5.1 时间对齐

论文定义历史 `a[t-7:t]`、未来 `a[t+1:t+8]`。仓库 loader 的固定窗口是 state `[-7..0]`、action `[0..7]`。两者只有在数据行 `t` 的 action `[0]` 定义为“观察 t 后下一条控制目标”时等价。若 action `[0]` 是同步 `a_t`，预处理必须向后平移，不能让 source/target 重叠一帧。

### 5.2 原始 LIBERO 为什么被拒绝

原始 LIBERO state 是 absolute measured EEF xyz + absolute rotation-vector + 2D gripper qpos；action 是 dimensionless OSC_POSE delta command + 1D gripper command。二者不在同一空间。

即使知道当前 `q[t]`，也只能精确构造第一步 controller goal。后续 action 以每一步真实 measured `q[t+k]` 为参考，受跟踪误差、碰撞、接触和限幅影响；用上一 goal 递推不是数据真值。旋转又必须按矩阵群复合，不能把三个 rotation-vector 分量逐维相加。

所以 stock LIBERO 在严格模式会 fail-fast。这是正确性保护，不是可直接训练的 LIBERO adapter。若要使用 LIBERO，必须另行实现：

1. 离线读取每个 future action 对应时刻的 measured EEF，生成版本化 absolute controller-goal canonical 列；或改用 actual future pose target。
2. gripper 单独作为 binary auxiliary。
3. 在线输出 absolute target 时，使用每控制子步最新 measured state 做 inverse-controller；第一版推荐 `n_action_steps=1` receding horizon。
4. 将上述 controller commit、scale、frame、旋转左/右乘和时序全部写入 contract。

另一条独立路线是 command-history A2A：训练窗口改成过去和未来均为实际下发的 OSC command，在线 buffer 也记录 issued command。它同空间且实现更直接，但语义是 commanded history，不是论文强调的 executed proprio feedback，也不是当前模型输入契约。

## 6. 数据准备流程

### 第一步：预处理 canonical 列

当前通用 exporter 的 canonicalizer 固定为 `identity_preprocessed`，意味着真正的机器人/控制器转换必须先完成。参考：

- `examples/A2A/canonical_modality.py`
- `examples/A2A/channel_specs.example.json`
- `examples/A2A/channel_contract.example.json`

复制示例并替换所有 `REPLACE_*` 值。不要只改 raw 列名。

### 第二步：导出严格窗口

```bash
python scripts/a2a/export_canonical_windows.py \
  --dataset-path <canonical_lerobot_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path <canonical_modality.py> \
  --contract <channel_contract.json> \
  --output outputs/a2a/canonical_windows.npz
```

NPZ 至少包含 `history/future/history_mask/future_mask`、`input_space`、完整 `contract_json`/hash 和 N/H/F/D。物理值仍未归一化。

### 第三步：审计

```bash
python scripts/a2a/audit_a2a_data.py \
  --input outputs/a2a/canonical_windows.npz \
  --output outputs/a2a/audit.json \
  --fail-on-errors
```

重点阅读：active nonfinite 是否为 0、constant channel、temporal jump、history→future 与 Gaussian→future 距离比。审计通过说明文件结构和统计证据合理，不证明机器人转换在现实中正确；转换仍需通过 controller replay/单元测试验证。

### 第四步：生成共享统计

```bash
python scripts/a2a/build_canonical_stats.py \
  --input outputs/a2a/canonical_windows.npz \
  --contract <channel_contract.json> \
  --output outputs/a2a/a2a_statistics.json
```

保存控制台打印的 channel contract SHA，训练参数必须提供同一个值。Processor 会验证 statistics 自身 SHA、version、input space、N/H/F/D、所有向量长度/finite/顺序、active/constant/count 和完整 contract。

## 7. 训练、checkpoint 与冻结策略

### 7.1 AE warmup

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <original_n1d7_or_full_a2a_checkpoint> \
  --dataset-path <canonical_lerobot_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path <canonical_modality.py> \
  --a2a-channel-specs-path <channel_specs.json> \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <contract_sha256> \
  --a2a-training-stage autoencoder \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_ae
```

`autoencoder` 只运行 trajectory encoder/decoder；VLM、condition pooler、FlowNet 和 auxiliary head 不参与。代码要求 `tune_projector=True`，并拒绝在这个阶段把 LLM/visual 标成可训练却实际绕过。

### 7.2 联合训练

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path outputs/a2a_ae/<checkpoint> \
  --dataset-path <canonical_lerobot_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path <canonical_modality.py> \
  --a2a-channel-specs-path <channel_specs.json> \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <contract_sha256> \
  --a2a-training-stage joint \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_joint
```

完整 A2A checkpoint 转 `joint` 时 pipeline 会把当前请求配置覆盖进 checkpoint config，避免 AE checkpoint 仍停留在 `autoencoder` stage。`flow_only` 只允许从完整 A2A checkpoint 开始，并冻结 encoder/decoder；不允许用原 N1.7 加随机 AE 冻结训练。

### 7.3 从原始 N1.7 迁移了哪些权重

原始 `Gr00tN1d7` 只提供 VLM backbone 和 VLLN allowlist；旧 action encoder、DiT 和 decoder 不会加载到 A2A 头。shape/type/未知 key 会形成明确迁移报告。完整 A2A resume 才恢复 trajectory AE、FlowNet、condition/aux head 与 processor contract。

### 7.4 训练噪声

默认仅训练时向有效 history 添加 `std=0.02` 噪声；`tau` 始终以 FP32 从连续 Uniform 采样，避免 BF16 把时间量化成 256 个值。默认 inference noise 为 0，所以推理确定性。论文不同噪声消融使用过不同标准差；若复现其多模态附录，需显式建立对应配置和随机 seed，不能把默认 0.02 视为所有论文主实验的统一固定值。

## 8. 在线推理

推荐客户端直接提供完整实际 history：

```python
observation = {
    "video": {"front": video_batch},
    "state": {"eef_pose_canonical": history_batch},  # [B, 8, D_state]
    "language": {"annotation.human.task_description": language_batch},
}
actions, info = policy.get_action(observation)
```

兼容旧客户端时，每次可发送 `[B,1,D]` 当前 measured state。policy 为每个 batch element 累积实际历史并输出 valid mask：

- `repeat_first_state`：首帧不足时左侧重复，但重复位 mask=0；模型不会把它当真实历史。
- `require_full_history`：未满 8 个真实样本直接拒绝。
- `timestamp` / `state_history_timestamps`：检查严格单调和外部完整历史。
- `a2a_max_time_gap_s`：拒绝丢帧造成的过长间隔。
- `policy.reset()`：新 episode、急停、任务切换必须调用。

PyTorch inference options 可临时传 `num_inference_steps=1/2/4/6`。TensorRT 的 solver steps 已在图中展开，运行时不能改变。

## 9. ONNX 与 TensorRT

### 9.1 导出

```bash
python scripts/deployment/export_onnx_n1d7_a2a.py \
  --model-path <a2a_checkpoint> \
  --output-dir outputs/a2a_onnx
```

图的 10 个输入是 backbone features/attention/image mask、history/mask、四种 head mask 和 embodiment id；`future_action_mask` 在 wrapper 内由四种互斥 head mask 求并集，避免被 ONNX DCE 删除后出现 10/11 参数错位。图固定 FP32、动态 batch 与 VLM sequence，禁止推理 history noise 和 current-state ablation。

默认导出检查：

1. `onnx.checker.check_model`；
2. 同一固定输入上 PyTorch 与 ONNX Runtime `rtol=atol=1e-4`；
3. 计算 ONNX SHA；
4. 绑定 checkpoint config+weight SHA、canonical statistics SHA、data contract SHA 和结构 contract SHA。

### 9.2 构建

```bash
python scripts/deployment/build_tensorrt_engine.py \
  --mode a2a_action_head \
  --onnx outputs/a2a_onnx/a2a_action_head.onnx \
  --engine outputs/a2a_engine/a2a_action_head.engine \
  --a2a-vl-sequence-length 512 \
  --a2a-max-batch 1
```

builder 固定与 strongly-typed ONNX 相符的 FP32，并拒绝 ONNX SHA 不匹配、缺 identity 或跳过数值 oracle 的图。构建后写 engine SHA 与实际 min/opt/max profile。

### 9.3 加载

`setup_tensorrt_engines(policy, engine_dir, mode="a2a_action_head")` 在导入 TensorRT 前验证 model type、H/F/D/L、solver steps、输入名/dtype、data contract、statistics、checkpoint 权重、ONNX identity 与 engine SHA。运行时每次按 engine 声明 dtype 转换，并为动态 batch/sequence 设置 tensor shape。

## 10. 如何证明实现和效果正确

### 10.1 软件正确性

至少要求：

- A2A modules shape/gradient/mask/padding/Euler oracle 测试；
- FM/AE/IC 公式、stage、finite 和每样本门禁测试；
- dataset 负索引/episode 边界测试；
- stats/contract/processor save-load 测试；
- history buffer/cold-start/timestamp/reset 测试；
- checkpoint 类型/迁移/恢复测试；
- ONNX/TRT 输入、dtype、dynamic shape、identity 路由测试；
- 原始 N1.7 模型与 processor 回归测试。

### 10.2 数据和训练正确性

1. 用很小数据把 AE overfit，查看 normalized 和物理单位重建误差。
2. 监控 `loss_fm/loss_ae/loss_ic_latent/loss_ic_action/loss_aux`。
3. 检查 `latent_std` 不趋近 0、pair distance 合理；需要时增加 per-dim std/分位数离线诊断。
4. 验证 history→future 距离小于 Gaussian→future；否则 A2A 的短运输假设不成立。
5. 在完全相同 observation/action chunk/训练预算下对比原 N1.7 4-step、原 N1.7 1-step、A2A 无 IC 和完整 A2A 1/2/4/6-step。
6. 至少 3 个 seed，报告成功率均值/方差、action smoothness、collision/limit violation、冷启动、视觉扰动、未见初态和 P50/P95 端到端延迟。

论文主表的 A2A 使用 6 steps；论文也显示充分训练后单步可以达到高质量。论文 action-head 延迟不能直接外推到包含 GR00T VLM 的端到端延迟，必须在目标硬件重新测量。

## 11. 常见失败与含义

- `Strict A2A requires explicit a2a_channel_specs`：没有证明历史与未来同物理空间。
- `Raw LIBERO ... OSC delta commands`：正尝试把原始 LIBERO 绝对 state 与 delta command 直接配对；必须先 canonicalize。
- `contract does not match expected`：通道顺序、控制器或 dataset provenance 漂移。
- `requires 8 real history steps`：`require_full_history` 冷启动尚未完成或 valid mask 含 padding。
- `history and future continuous masks must describe the same channels`：shared encoder 两侧的连续语义不一致。
- `generation masks must be zero outside future_action_mask`：padding 位置被错误分给 head。
- `active constant canonical channels`：统计中有零方差有效通道；先检查数据或显式移出 continuous latent。
- `checkpoint SHA ... does not match`：TensorRT engine 来自不同 A2A 权重，即使模型 shape 一样也不允许加载。
- `fixed number of Euler steps`：TensorRT 图的步数不可运行时覆盖，需要重新导出。

## 12. 当前交付状态

已经完成的通用能力是：论文核心潜空间 A2A 模型、严格 canonical 数据链、训练/checkpoint/policy、ONNX/TensorRT 合约和原模型隔离。

没有替用户猜测或伪造的部分是：任意具体机器人（特别是原始 LIBERO）的不可逆控制器转换、训练完成权重和任务效果。对已经预处理成同一 absolute canonical 空间的机器人数据，代码链路可进入训练与评估；对 stock LIBERO，严格模式会有意阻断，直到提供可验证的离线 canonicalizer 与在线 inverse-controller adapter。
