# GR00T 项目的三种 Action 生成方式与完整使用手册

## 1. 项目目前支持哪三种方式

### 1.1 原始 GR00T N1.7

```text
model_type = Gr00tN1d7
```

计算过程：

```text
图像 + 语言 + 当前状态
        -> GR00T VLM features
高斯动作噪声 + flow time + VLM condition
        -> 原始大 DiT/Transformer Action Head
        -> 未来 action chunk
```

它学习的是 noise-to-action velocity field。推理起点是随机高斯动作张量。

### 1.2 潜空间 A2A-MLP

```text
model_type = Gr00tN1d7A2A
a2a_flow_backbone = mlp
```

计算过程：

```text
最近 8 步真实执行历史 -> 共享 Conv encoder -> z0 [B,512]
未来 8 步示范动作       -> 同一 Conv encoder -> z1 [B,512]（仅训练）
VLM features             -> condition pooler -> c [B,512]
z_tau + tau + c           -> 4-block AdaLN-MLP -> latent velocity
z0 经 Euler 积分          -> z1_hat -> decoder -> 未来动作
```

它学习 history-latent 到 future-latent 的速度场。推理起点是历史动作 latent，不是噪声。

### 1.3 潜空间 A2A-DiT

```text
model_type = Gr00tN1d7A2A
a2a_flow_backbone = dit
a2a_strict_paper_architecture = false
```

计算过程与 A2A-MLP 的数学目标相同，但 latent 保留为时间 token：

```text
最近 8 步历史 -> token encoder -> z0 [B,8,C]
未来 8 步动作 -> 同一 encoder -> z1 [B,8,C]
z_tau + tau + VLM condition
        -> temporal AdaLN-DiT
        -> token velocity [B,8,C]
        -> Euler -> token decoder -> 未来动作
```

默认 `C=256`；公平 512 元素预算对照用 `C=64`，因为 `8×64=512`。

## 2. 先选择哪种方式

| 目标 | 建议 |
|---|---|
| 使用官方原模型或保持已有行为 | `Gr00tN1d7` |
| 复现论文核心潜空间 A2A | `Gr00tN1d7A2A + mlp` |
| 对比更大时序 Transformer velocity field | `Gr00tN1d7A2A + dit` |
| 第一次建立可靠 A2A 基线 | 先 MLP，再 DiT-64，最后 DiT-256 |

MLP checkpoint 与 DiT checkpoint 不能互换。三组实验必须使用不同输出目录。

## 3. 离线电脑还需要准备什么

本交付包只包含源码和 patch，不包含：

- Python/CUDA/PyTorch 环境；
- NVIDIA/Cosmos/GR00T 模型权重；
- Hugging Face cache；
- 机器人数据集；
- 训练得到的 A2A checkpoint；
- TensorRT/CUDA runtime。

离线前应从联网机器复制：

1. 完整项目和已安装好的虚拟环境，或所有 wheel/conda 包；
2. 原始 `nvidia/GR00T-N1.7-3B` 与 `nvidia/Cosmos-Reason2-2B` 本地 checkpoint/cache；
3. canonicalized 数据集、channel specs、contract、NPZ 和 statistics；
4. 如果需要部署，复制 ONNX Runtime、TensorRT 和对应 CUDA 版本安装包。

## 4. A2A 数据为什么不能直接使用任意 state/action

A2A continuous encoder 假定 history 和 future 是同一个物理量。例如两侧都应是：

```text
[x, y, z, rotation_6d]，单位和坐标系相同，均表示 absolute controller goal
```

不能把以下两者直接放在一起：

```text
history = absolute measured EEF pose
future  = dimensionless delta controller command
```

即使维度相同，这也不是同一动作空间。严格模式要求显式 channel spec、semantic metadata、canonical statistics 和 contract hash。

## 5. 第一步：生成 canonical 数据列

参考文件：

```text
examples/A2A/canonical_modality.py
examples/A2A/channel_specs.example.json
examples/A2A/channel_contract.example.json
```

### 5.1 modality 配置

连续历史和未来应指向已经预处理好的同空间列。例如：

```python
state = ModalityConfig(
    delta_indices=[-7, -6, -5, -4, -3, -2, -1, 0],
    modality_keys=["eef_pose_canonical"],
)
action = ModalityConfig(
    delta_indices=[0, 1, 2, 3, 4, 5, 6, 7],
    modality_keys=["eef_goal_canonical"],
)
```

### 5.2 channel specs

文件最外层 key 是 embodiment tag：

```json
{
  "new_embodiment": [
    {
      "action_key": "eef_goal_canonical",
      "source_state_key": "eef_pose_canonical",
      "kind": "continuous",
      "canonical_format": "xyz_rot6d",
      "source_format": "xyz_rot6d",
      "target_format": "xyz_rot6d",
      "source_unit": "meter_unitless",
      "target_unit": "meter_unitless",
      "source_frame": "world",
      "target_frame": "world",
      "dim": 9
    },
    {
      "action_key": "gripper_command",
      "kind": "binary",
      "canonical_format": "scalar",
      "source_format": "binary",
      "target_format": "binary",
      "source_unit": "unitless",
      "target_unit": "unitless",
      "source_frame": "controller",
      "target_frame": "controller",
      "dim": 1
    }
  ]
}
```

只有 `continuous` 进入共享 A2A latent。`binary/regression/categorical` 使用 auxiliary head；`unsupported` 会 fail-fast。

### 5.3 channel contract

contract 必须记录：

- dataset immutable revision/fingerprint；
- source schema/modality hash；
- canonicalizer name/version/code hash；
- controller type/version；
- translation/rotation scale；
- `control_delta`、frame、旋转左乘/右乘；
- observation/action 时间对齐；
- target 是 controller goal 还是 achieved pose；
- gripper mapping；
- channel 顺序和范围。

## 6. 第二步：导出未归一化 canonical 窗口

在项目根目录运行：

```bash
python scripts/a2a/export_canonical_windows.py \
  --dataset-path <canonical_lerobot_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --contract <channel_contract.json> \
  --output outputs/a2a/canonical_windows.npz
```

输出 NPZ 包含：

```text
history       [N,8,D]
future        [N,8,D]
history_mask  [N,8,D]
future_mask   [N,8,D]
N/H/F/D metadata
contract_json
contract_sha256
input_space = canonical_physical_unnormalized
```

exporter 不执行机器人控制器变换；它要求输入列已经 canonicalized。

## 7. 第三步：数据审计

```bash
python scripts/a2a/audit_a2a_data.py \
  --input outputs/a2a/canonical_windows.npz \
  --output outputs/a2a/audit.json \
  --fail-on-errors
```

必须重点确认：

- active nonfinite count 为 0；
- 每个 continuous channel 都有 history 和 future 样本；
- 没有异常常量通道；
- temporal jump 没有明显单位错误；
- history-to-future 距离相对 Gaussian-to-future 有合理优势；
- N/H/F/D 与 contract 一致。

审计通过只说明结构和统计合理；controller 变换仍需 replay/oracle 验证。

## 8. 第四步：生成共享 statistics

```bash
python scripts/a2a/build_canonical_stats.py \
  --input outputs/a2a/canonical_windows.npz \
  --contract <channel_contract.json> \
  --output outputs/a2a/a2a_statistics.json
```

history 和 future 使用同一套统计量。不要分别归一化，否则两侧不再处于相同坐标尺度。

记录命令打印的 `contract_sha256`。训练时必须传入同一个值。

## 9. 训练 A2A-MLP

### 9.1 推荐：先做 AE warmup

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <local_original_n1d7_checkpoint> \
  --dataset-path <canonical_lerobot_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path <channel_specs.json> \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <contract_sha256> \
  --a2a-flow-backbone mlp \
  --a2a-training-stage autoencoder \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_mlp_ae
```

### 9.2 联合训练

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path outputs/a2a_mlp_ae/<checkpoint> \
  --dataset-path <same_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path <same_channel_specs.json> \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <same_contract_sha256> \
  --a2a-flow-backbone mlp \
  --a2a-training-stage joint \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_mlp_joint
```

`flow_only` 只能从已训练的完整 A2A checkpoint 开始；不能冻结随机 encoder/decoder。

## 10. 训练 A2A-DiT

DiT 必须显式关闭论文核心架构锁：

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <same_original_n1d7_checkpoint> \
  --dataset-path <same_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path <same_channel_specs.json> \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
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
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_dit_large
```

公平预算版只需改：

```bash
--a2a-dit-token-dim 64
```

如果进行 AE warmup，MLP 与 DiT 要分别 warmup。不要交叉加载 AE checkpoint。

## 11. 训练时看哪些日志

Trainer 会输出：

```text
a2a/loss_fm
a2a/loss_ae
a2a/loss_ic
a2a/loss_ic_latent
a2a/loss_ic_action
a2a/loss_aux
a2a/latent_std
a2a/latent_pair_distance
```

需要警惕：

- `loss_ae` 长期不降：canonical 数据、decoder 或 mask 有问题；
- `latent_std -> 0`：latent collapse；
- `latent_pair_distance -> 0` 且动作不对：encoder 失去区分能力；
- FM 很低但 closed-loop 差：不能只依赖 normalized latent loss；
- DiT loss 更低但延迟/成功率没有改善：容量增加未转化为任务收益。

## 12. 推理时输入什么

A2A policy 支持两种形式。

### 12.1 每次只提供当前真实状态

state shape 为 `[B,1,D_state]`。Policy 内部逐步积累 8 帧历史：

```python
action, info = policy.get_action(
    observation,
    options={"timestamp": current_time_seconds},
)
```

每个 episode 开始前必须：

```python
policy.reset()
```

### 12.2 外部直接提供完整历史

state shape 为 `[B,8,D_state]`，并提供严格递增时间戳：

```python
action, info = policy.get_action(
    observation_with_full_history,
    options={"state_history_timestamps": timestamps},
)
```

不要把模型上一次预测的 future action 直接写入 history。history 必须代表实际执行/测量结果。

## 13. 刚开始不足 8 帧如何推理

### `repeat_first_state`

- 用第一帧数值补齐到 8 帧；
- 补齐帧的 valid mask 为 0；
- encoder/DiT 知道哪些位置是假历史；
- 可以从第一帧开始输出动作。

### `require_full_history`

- 前 7 次只积累真实状态；
- 不足 8 帧直接抛出错误；
- 第 8 帧后才开始生成动作。

实际机器人通常可先静止采集 8 帧，再启用控制，以减少冷启动分布偏差。

## 14. A2A 推理内部发生什么

MLP 和 DiT 共用以下逻辑：

```text
1. processor 构造并共享归一化 history_action_canonical
2. encoder 得到 z0
3. VLM pooler 得到 condition c
4. z = z0
5. 对每个 Euler step：
       tau = step / num_steps
       z = z + dt * velocity_field(z, tau, c)
6. decoder 把最终 z 还原为连续未来轨迹
7. auxiliary head 生成 binary/regression/categorical 通道
8. 按 mask 合并并反归一化为控制器输出
```

默认 `num_inference_steps=1`。可以在推理 `options` 中临时传 `num_inference_steps` 做 1/2/4/6 步对照。

## 15. 原始 GR00T 与两种 A2A 的公平对比

必须固定：

- 相同原始 VLM base checkpoint；
- 相同数据划分和图像增强；
- 相同 VLM 冻结策略；
- 相同 optimizer、学习率、global batch、训练步数；
- 相同随机种子集合；
- A2A 两组使用相同 canonical stats/contract；
- 相同 action execution horizon 和控制频率。

建议三组：

```text
Baseline：原始 Gr00tN1d7
A2A-MLP：512 vector latent
A2A-DiT-64：8x64 token latent（相同元素预算）
A2A-DiT-256：8x256 token latent（容量上限）
```

严格来说这是四组；如果只做三组，优先选择 Baseline、A2A-MLP、A2A-DiT-64。

## 16. 部署流程

### 16.1 导出 ONNX

使用：

```text
scripts/deployment/export_onnx_n1d7_a2a.py
```

导出的是固定 Euler step 的 fused FP32 A2A Action Head，并生成 `a2a_export_metadata.json`。metadata 绑定 checkpoint、backend、statistics、contract 和 ONNX SHA。

### 16.2 构建 TensorRT engine

使用 `build_tensorrt_engine.py` 的 `a2a_action_head` mode。A2A 路径强制 FP32 contract，并为 `batch` 与 `vl_sequence` 创建显式 profile。

### 16.3 运行时安装 engine

`trt_model_forward.py` 路由到 `a2a_trt_model_forward.py`。运行前会核对 engine metadata、checkpoint identity、A2A 架构、数据 contract 和 statistics。

当前 CPU 环境已验证导出和入口逻辑，但没有实际 CUDA/TensorRT engine benchmark；部署电脑需要单独完成 GPU 验收。

## 17. 常见错误

| 报错/现象 | 原因 | 处理 |
|---|---|---|
| 缺少 channel specs | strict A2A 不允许自动猜映射 | 提供显式 JSON |
| contract SHA mismatch | stats、数据或配置来自不同转换版本 | 重新生成或使用正确 artifact |
| MLP checkpoint 无法载入 DiT | action head 架构不同 | 从同一原始 base 分别训练 |
| `require_full_history` 拒绝推理 | 真实历史不足 8 帧 | 继续采集或使用 repeat-first |
| timestamp non-monotonic | 外部时间戳倒退/重复 | 每个 batch 保持严格递增，episode reset |
| raw LIBERO 被拒绝 | absolute state 与 delta command 不同空间 | 实现 controller-aware canonicalizer |
| DiT 显存明显增加 | 8×256 latent + attention 容量更大 | 先用 token_dim=64 |
| loss 正常但动作不正确 | 物理语义、时间对齐或 inverse-controller 错 | 做 controller replay 和闭环验证 |
