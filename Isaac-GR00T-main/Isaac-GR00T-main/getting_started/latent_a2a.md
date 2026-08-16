# GR00T N1.7 潜空间 Action-to-Action Flow Matching

本文说明仓库内 `Gr00tN1d7A2A` 变体的适用范围、数据准备、训练、推理与部署流程。默认 `a2a_flow_backbone="mlp"` 复用 GR00T N1.7 的视觉语言主干，用论文式 512 维轨迹潜空间 A2A 头替换原始的高斯噪声到动作 DiT 头；原始 `Gr00tN1d7` 保持不变。仓库还提供不覆盖默认实现的时间序列 token DiT 对照后端，结构、配置与公平对比方法见 [`潜空间A2A_DiT对比实现与使用.md`](潜空间A2A_DiT对比实现与使用.md)。

## 1. 已实现的方法

固定的论文核心迁移配置为：

- 历史实际 proprio 轨迹 `q[t-7:t]`：8 步；
- 未来示范动作 `u[t:t+7]`：8 步；
- 共享轨迹编码器：3 层 Conv1d，通道 `(128, 256, 512)`，kernel 5；
- 轨迹潜变量：512 维；
- 条件网络：4 个 AdaLN residual MLP block，MLP ratio 4；
- 解码器：4 个 residual MLP block；
- 训练时间 `tau ~ Uniform(0,1)`；
- 推理从 `z0 = E(history)` 开始，默认一步 Euler，不创建高斯动作初值；
- 总损失：`L_FM + 0.5 L_AE + L_IC + L_aux`，其中 `L_IC = L1(z_hat1,z1) + 0.5 L1(D(z_hat1),future)`。

这里的“论文核心迁移”指潜空间 A2A 的源/目标、共享编码器、直线 flow matching、AE/IC 损失和历史潜变量起点与论文一致；它不是论文实验网络的逐比特复刻。论文使用 ResNet18 和 8 帧视觉历史，本项目按迁移方案保留 GR00T VLM、语言条件与当前/原配置视觉输入。`(128,256,512)` 中间卷积宽度、MLP ratio、激活与归一化也是仓库实现选择，因为论文没有公开这些细节。

对于纯连续动作，`L_aux=0`，总损失与论文公式完全相同。binary/categorical/regression auxiliary head 是为 GR00T 多 embodiment 和混合动作增加的工程扩展；论文把混合连续/离散动作支持列为后续方向。

历史和未来严格使用同一个 `ActionTrajectoryEncoder`。训练路径中的一致性积分保持可微，梯度能够回到编码器、解码器、FlowNet 和条件池化器。离散/不具备 executed-proprio 对应关系的通道不会混入连续潜空间，而是进入 auxiliary regression、binary 或 categorical head。

推荐按三阶段训练：

1. `autoencoder`：只训练轨迹编码器/解码器，先验证动作轨迹可重建；
2. `joint`：联合优化 FM、AE、IC 与辅助头；
3. `flow_only`：仅用于已经存在完整 A2A checkpoint 的后续实验，代码拒绝从随机 AE 开始冻结训练。

## 2. 最重要的数据前提

连续 A2A 的历史和未来必须是同一种物理量，并具有相同格式、单位、坐标系和时间定义。维度相同并不能证明这一点。

### 原始 LIBERO 数据不能直接训练本实现

仓库中的原始 LIBERO 表示为：

- state：绝对测量的 EEF `xyz`、绝对 rotation-vector（字段虽名为 roll/pitch/yaw，但并非 Euler）以及 2 维 gripper qpos；
- action：归一化的 6 维 OSC_POSE 增量控制命令以及 1 维 gripper command。

两侧不在同一空间。仅用 `q[t]` 和未来 action chunk 只能精确得到第一步 controller goal；后续增量命令以每一步当时的实际测量状态为参考，受到闭环跟踪、接触和限幅影响，不能用前一步 goal 无损递推。

因此严格配置会拒绝以下做法：

- 因为 state/action 维度相等就自动标成 continuous；
- 把 raw LIBERO 的绝对位置与 OSC delta 共用一套统计；
- 把三个 rotation-vector 分量当作可独立相加的 Euler 角；
- 将 2 维 gripper qpos 与 1 维 binary command 当作可逆连续映射。

可行的数据路线有两种：

1. 离线生成新的、已经对齐的 canonical 列，例如历史与未来都采用 absolute `xyz+rot6d` controller target，并为每个未来 action 使用同一时刻的 measured state 完成转换；gripper 单独作为 binary auxiliary。
2. 另做 command-history A2A：历史和未来均使用过去/未来实际下发的 OSC command。这是另一种方法语义，不是当前实现的 executed-proprio history，需新增 past-action 数据窗口与在线 command buffer。

若使用 absolute canonical EEF target，运行时还必须逐控制子步使用最新测量状态做逆控制转换。现有 open-loop multistep wrapper 不提供这种反馈；第一版应使用 `n_action_steps=1` 的 receding-horizon，或在环境 wrapper 中实现经过版本锁定的逐步 adapter。

## 3. 数据契约与 canonical 窗口

项目提供三个示例：

- `examples/A2A/canonical_modality.py`：已经预处理为 canonical 列的 modality 注册；
- `examples/A2A/channel_specs.example.json`：模型 processor 使用的显式通道分类；
- `examples/A2A/channel_contract.example.json`：统计与控制器 provenance 契约。

必须替换示例中的所有 `REPLACE_*` 值。契约至少绑定：不可变数据集 revision/fingerprint、原始 schema hash、canonicalizer 版本与代码 hash、控制器类型/版本/scale/frame、rotation composition、目标定义和 observation/action 时间对齐。

当前通用导出器只支持 `identity_preprocessed`：数据列必须在进入导出器之前已经完成 canonicalization。它不会也不能根据字符串声明替用户转换控制器语义。

```bash
python scripts/a2a/export_canonical_windows.py \
  --dataset-path <preprocessed_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --contract examples/A2A/channel_contract.json \
  --output outputs/a2a/canonical_windows.npz
```

导出器直接遍历完整 episode 的严格 anchor，使用 state `[-7..0]` 和 action `[0..7]`，禁止 padding、负索引回绕、跨 episode 读取、非有限值、重复 anchor 及错误 mask。输出仍是未归一化物理 canonical 值，不经过模型 Processor。

索引语义必须由数据契约明确：论文写作 `a[t-7:t] -> a[t+1:t+8]`；本项目采用 LeRobot/GR00T 的 anchor 约定 `state[-7:0] -> action[0:7]`。这只有在数据行 `t` 的 `action[0]` 表示“观察 `t` 后将执行的下一控制目标”时等价。若你的数据把该列定义成同步的 `a_t`，必须在预处理时平移 action，避免历史与未来重叠一帧；自由文本 `time_alignment` 不能替代对数据生成逻辑的验证。

先审计，再构建历史/未来共享统计：

```bash
python scripts/a2a/audit_a2a_data.py \
  --input outputs/a2a/canonical_windows.npz \
  --output outputs/a2a/audit.json \
  --fail-on-errors

python scripts/a2a/build_canonical_stats.py \
  --input outputs/a2a/canonical_windows.npz \
  --contract examples/A2A/channel_contract.json \
  --output outputs/a2a/a2a_statistics.json
```

统计脚本合并 history 与 future 的连续通道并使用同一组尺度；inactive padding 保持中性统计。输出会打印 channel contract SHA-256 和完整 statistics SHA-256。训练时必须把前者作为 `--a2a-expected-contract-sha256`，以避免通道顺序、控制器版本或数据 provenance 漂移。

## 4. 训练

标准 finetune 入口已支持 A2A。以下命令中的 base checkpoint 可以是原始 `Gr00tN1d7`（只迁移 backbone 与 VLLN allowlist），也可以是完整 `Gr00tN1d7A2A` checkpoint（严格恢复）。

AE warmup：

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path <original_n1d7_or_a2a_checkpoint> \
  --dataset-path <preprocessed_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path examples/A2A/channel_specs.json \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <printed_contract_sha256> \
  --a2a-training-stage autoencoder \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_ae
```

联合训练：

```bash
python gr00t/experiment/launch_finetune.py \
  --model-type Gr00tN1d7A2A \
  --base-model-path outputs/a2a_ae/<checkpoint> \
  --dataset-path <preprocessed_dataset> \
  --embodiment-tag new_embodiment \
  --modality-config-path examples/A2A/canonical_modality.py \
  --a2a-channel-specs-path examples/A2A/channel_specs.json \
  --a2a-canonical-statistics-path outputs/a2a/a2a_statistics.json \
  --a2a-expected-contract-sha256 <printed_contract_sha256> \
  --a2a-training-stage joint \
  --global-batch-size 32 \
  --state-dropout-prob 0 \
  --output-dir outputs/a2a_joint
```

注意：A2A 历史不会经过旧的 relative-action 变换；`launch_finetune.py` 会为 A2A 自动关闭该变换。完整 checkpoint 保存 config、processor、通道 spec 与 canonical statistics。原始 N1.7 checkpoint 的迁移只允许明确的 backbone/VLLN 键，并生成迁移报告；未知或错误模型类型会失败。

上例显式使用论文 Appendix A.2 的 global batch size 32；学习率、训练轮数、优化器、图像增强和 VLM 冻结策略仍应按 GR00T 迁移实验单独记录，不能因 A2A 核心参数一致就假设等同于论文实验。

## 5. 在线推理

优先让客户端每次请求直接发送 `[B,8,D]` 的实际 measured proprio history。兼容旧客户端时，policy 可接收 `[B,1,D]` 当前反馈并用 ring buffer 累积；它从不把预测动作当成 executed history。

可选 inference options：

- `timestamp`：单帧实际反馈的时间戳；
- `state_history_timestamps`：外部完整 8 帧历史的时间戳；
- `num_inference_steps`：PyTorch 路径临时覆盖 Euler 步数；TensorRT 固定在导出时的步数。

`policy.reset()` 会在新 episode、急停或任务切换时清空 history。`a2a_max_time_gap_s` 可拒绝过长采样间隔。冷启动模式：

- `repeat_first_state`：重复首帧，但前置位置的 valid mask 为 0；
- `require_full_history`：未收到 8 个真实样本前直接拒绝推理。

## 6. ONNX 与 TensorRT

A2A 使用独立的 10 输入 fused action-head 图，不覆盖原始 N1.7 部署脚本。导出要求推理历史噪声为 0，因此图内没有随机算子：

```bash
python scripts/deployment/export_onnx_n1d7_a2a.py \
  --model-path <a2a_checkpoint> \
  --output-dir outputs/a2a_onnx

python scripts/deployment/build_tensorrt_engine.py \
  --mode a2a_action_head \
  --onnx outputs/a2a_onnx/a2a_action_head.onnx \
  --engine outputs/a2a_engine/a2a_action_head.engine \
  --a2a-vl-sequence-length 512 \
  --a2a-max-batch 1
```

专用构建入口固定 FP32，与当前 strongly-typed ONNX 一致，并写入 engine SHA-256、输入 profile 和模型 contract metadata。标准 `setup_tensorrt_engines(..., mode="a2a_action_head")`、benchmark、verify 和 rollout mode 均可选择该路径；加载时会在导入 TensorRT 之前核对模型类型、H/F、D、Euler steps、10 输入、contract 与 engine hash。

导出默认还运行 ONNX checker 和同一固定输入上的 PyTorch/ONNX Runtime 数值对照（`rtol=atol=1e-4`）；metadata 会绑定 checkpoint 权重/config SHA、canonical statistics SHA、数据 contract SHA 和 ONNX SHA。TensorRT 构建及加载会再次校验这些值，防止把形状相同但权重或数据语义不同的 engine 绑定到当前 checkpoint。只有在调试缺少 ONNX Runtime 的环境时才使用 `--no-verify-export`，而专用 TensorRT builder 会拒绝这种未验证图。

## 7. 验收建议

至少完成以下检查再比较闭环成功率：

1. 数据审计无错误，history→future 距离显著小于 Gaussian→future；
2. AE tiny-overfit，物理单位重建误差低于控制器允许误差，latent std 不坍缩；
3. `L_FM/L_AE/L_IC` 正常下降，一步与 2/4/6 步结果随积分精度合理变化；
4. 原始 `Gr00tN1d7` 回归测试继续通过；
5. checkpoint save/load 输出一致；
6. policy reset、cold-start mask、时间戳和丢帧门禁正确；
7. PyTorch 与 ONNX/TensorRT 固定输入误差满足阈值；
8. 使用至少 3 个 seed，对比原 N1.7 四步、原 N1.7 一步、latent A2A 无 IC、完整 latent A2A 的成功率、平滑性、延迟和鲁棒性。

本实现提供的是论文方法的模型、数据契约、训练、策略与部署基础设施。它不会替特定机器人猜测不可逆的 state/action 物理转换；机器人专属 canonicalizer 和运行时 inverse-controller adapter 必须由数据与控制器真实定义完成并纳入契约。

代码级测试通过只能证明公式、张量形状、梯度、边界、checkpoint 和部署合约按预期运行，不能替代论文效果复现。要声称效果复现，仍需真实 canonical 数据上的 AE tiny-overfit、loss/latent 诊断、至少 3 个 seed 的闭环成功率、1/2/4/6 步对照和端到端延迟测量。论文 Table 1 的主表采用 6 步；默认 1 步是论文展示的低延迟目标配置。默认推理历史噪声为 0，保持确定性；若复现论文附录的多模态噪声消融，需要显式把 inference noise 配成相应实验值，不能把训练噪声 `0.02` 当成唯一论文固定超参。
