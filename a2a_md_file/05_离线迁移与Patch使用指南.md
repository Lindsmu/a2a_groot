# GR00T 潜空间 A2A：无网络电脑迁移与 Patch 使用指南

## 1. 迁移包的基线

统一 patch 不是对任意 GR00T 版本通用。它精确基于：

```text
NVIDIA/Isaac-GR00T
GR00T N1.7 General Release
commit 1a1837f20538b7d7e21f977a11a5aee14f99803c
```

本地原项目是 ZIP 解压目录，没有 `.git`。我通过未修改的 `pyproject.toml`、`uv.lock`、`LICENSE` Git blob 与 NVIDIA 官方历史逐项匹配，定位到上述提交，再以官方源码压缩包建立基线。

## 2. 交付文件的关系

```text
统一 patch：看“原版每一行如何变成当前实现”
offline_overlay：得到每个最终文件的完整内容
manifest：验证每个文件和基线/最终 SHA
逐行源码定位：不依赖 IDE 查看精确行号
apply_offline_overlay.ps1：无 Git 时自动备份、复制、验证
```

两种安装方法最终内容等价；已用归一化 LF SHA 验证 52/52 文件一致。

## 3. 方法 A：使用 Git patch（最适合审阅）

### 3.1 先备份

先完整复制项目目录，或至少保存本地尚未提交的改动。不要在有重要未保存修改的目录中直接应用。

### 3.2 进入项目根目录

项目根目录必须直接包含：

```text
README.md
gr00t/
scripts/
tests/
```

### 3.3 检查 patch 能否干净应用

Windows PowerShell：

```powershell
git apply --check "<离线包>\GR00T_N1.7_General_Release_to_Latent_A2A_MLP_DiT.patch"
```

Linux：

```bash
git apply --check /path/to/GR00T_N1.7_General_Release_to_Latent_A2A_MLP_DiT.patch
```

如果 `--check` 失败，不要加 `--reject` 或盲目强制。先确认基线提交；如果项目已经有本地修改，使用 overlay 逐文件合并或查看 patch 冲突位置。

### 3.4 应用

```powershell
git apply --whitespace=nowarn "<离线包>\GR00T_N1.7_General_Release_to_Latent_A2A_MLP_DiT.patch"
```

`--whitespace=nowarn` 只隐藏新增 `__init__.py` 文件末尾空行提示，不改变代码语义。

### 3.5 验证修改状态

```powershell
git status --short
git diff --stat
```

预期 patch 范围：

```text
52 files changed
34 added files
18 modified files
9813 insertions
44 deletions
```

不同 Git 行尾配置可能显示 CRLF/LF 差异，但 `NormalizedLF_SHA256` 应一致。

## 4. 方法 B：无 Git 的 offline overlay（最稳妥）

### 4.1 自动安装

在 PowerShell 中运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "<离线包>\apply_offline_overlay.ps1" `
  -ProjectRoot "D:\path\to\Isaac-GR00T-main"
```

脚本会：

1. 检查 `ProjectRoot` 不是磁盘根目录且包含 `gr00t/`；
2. 对 `offline_overlay` 自身做 52 个 SHA 校验；
3. 对 modified 文件检查基线 normalized SHA；
4. 发现本地漂移时停止，不覆盖；
5. 在项目同级创建带时间戳的 backup；
6. 复制新增/修改文件；
7. 再次校验 52 个最终 SHA；
8. 在 backup 中保存 `NEW_FILES_TO_REMOVE_ON_ROLLBACK.txt`。

### 4.2 预演，不写文件

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "<离线包>\apply_offline_overlay.ps1" `
  -ProjectRoot "D:\path\to\Isaac-GR00T-main" `
  -WhatIf
```

### 4.3 项目已有本地改动

默认脚本会报 `Local drift detected`。推荐做法：

1. 打开统一 patch，定位对应文件/hunk；
2. 用 `offline_overlay` 中的最终文件做三方对比；
3. 手动合并本地逻辑和 A2A 逻辑；
4. 运行测试。

只有确认愿意由 overlay 覆盖并且已经备份，才使用：

```powershell
... -Force
```

### 4.4 手工复制

如果 PowerShell 不可用，可把 `offline_overlay/` 下的目录树合并到项目根目录。不要把 `offline_overlay` 目录本身套在项目里面；应保持例如：

```text
offline_overlay/gr00t/model/modules/a2a_latent.py
          ->
ProjectRoot/gr00t/model/modules/a2a_latent.py
```

手工复制前按 `IMPLEMENTATION_FILE_MANIFEST.csv` 备份 18 个 modified 文件。

## 5. 回滚 overlay

自动脚本创建的 backup 包含所有被覆盖的旧文件。回滚步骤：

1. 把 backup 中的目录树复制回项目根目录；
2. 打开 `NEW_FILES_TO_REMOVE_ON_ROLLBACK.txt`；
3. 逐项确认后删除这些新增文件；
4. 重新运行原始 GR00T 回归测试。

不要对项目根目录做递归删除。只处理清单中的精确文件。

## 6. 文件完整性验证

### 6.1 使用 manifest

`IMPLEMENTATION_FILE_MANIFEST.csv` 字段：

| 字段 | 含义 |
|---|---|
| `Path` | 相对项目根目录路径 |
| `ChangeType` | Added 或 Modified |
| `Lines/Bytes` | 最终文件规模 |
| `SHA256` | 最终文件原始字节 SHA |
| `NormalizedLF_SHA256` | CRLF/LF 统一后的最终 SHA |
| `Baseline_SHA256` | 官方基线原始字节 SHA |
| `Baseline_NormalizedLF_SHA256` | 官方基线行尾归一化 SHA |

跨 Windows/Linux 建议比较 normalized SHA。

### 6.2 为什么同时提供两种 SHA

Git 在 Windows 可能把 LF checkout 成 CRLF，文件功能不变但原始字节 SHA 不同。normalized SHA 先把 CRLF 转成 LF，再计算 UTF-8 SHA，因此用于跨平台内容一致性。

## 7. 应用后最低测试集

在已经装好依赖的项目环境中运行：

```bash
pytest \
  tests/gr00t/model/test_a2a_latent.py \
  tests/gr00t/model/test_a2a_action_head.py \
  tests/gr00t/model/test_a2a_processor.py \
  tests/gr00t/data/test_a2a_dataset.py \
  tests/gr00t/data/test_a2a_data_audit.py \
  tests/gr00t/data/test_a2a_canonical_export.py \
  tests/gr00t/policy/test_a2a_history.py \
  tests/gr00t/deployment/test_a2a_artifacts.py \
  tests/gr00t/experiment/test_trainer_a2a_logging.py \
  tests/gr00t/configs/test_base_config_safe_yaml.py \
  tests/scripts/deployment/test_a2a_dit_onnx.py \
  tests/scripts/deployment/test_a2a_strict_observation.py \
  tests/scripts/deployment/test_a2a_trt_entrypoints.py -q
```

预期：

```text
109 passed
```

原 N1.7 回归：

```bash
pytest \
  tests/gr00t/model/test_action_head.py \
  tests/gr00t/model/test_gr00t_processor.py \
  tests/gr00t/model/test_action_horizon_validation.py \
  tests/gr00t/model/test_model_forward.py -q
```

预期：

```text
38 passed
```

静态检查：

```bash
ruff check .
```

预期：

```text
All checks passed!
```

## 8. 没有 ONNX Runtime/TensorRT 怎么办

- 缺 ONNX/ONNX Runtime：先运行除 `test_a2a_dit_onnx.py` 外的测试；不能因此声称 ONNX 路径已在该电脑验证。
- 缺 TensorRT/CUDA：A2A TRT helper 可以被导入，但不能构建/运行真实 engine；保留 CPU/PyTorch 推理。
- 完全离线时不能临时下载依赖，必须提前复制 wheel/环境。

## 9. patch 中包含的共享兼容改动

为了让 patch 精确复现当前通过测试的最终文件，除 A2A 本身外还包含两个触及同一文件的共享兼容变化：

1. `README.md` 的本地 Hugging Face CLI 命令更新；
2. `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` 的 tensor image parity 修复，以及对应 `test_vlm_image_input_parity.py`。

它们不改变 A2A Flow Matching 数学，但属于当前最终项目状态，并已纳入 manifest 和 patch。

## 10. 不要只复制核心网络文件

只复制 `a2a_latent.py` 和 action head 会留下以下缺口：

- config 无法恢复 A2A 子类；
- dataset 可能负索引回绕；
- processor 没有共享 statistics 和 semantic contract；
- online cold-start/history 不正确；
- checkpoint 会错误迁移；
- ONNX/TRT metadata 不绑定权重和数据；
- 标准 CLI 无法选择 DiT。

要复现当前通过测试的实现，应应用完整 52 文件 patch/overlay。
