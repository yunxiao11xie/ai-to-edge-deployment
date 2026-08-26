---
tags:
  - RK3576
  - Qwen3.5
  - RKLLM
  - 实战
aliases:
  - Qwen3.5-2B RK3576部署
---

# Qwen3.5-2B 在 RK3576 上的完整可复现实战

> 这不是一份“命令收藏”，而是一份带输入、输出、版本、检查点和失败边界的实验手册。当前已验证的是 **Qwen3.5-2B 语言模型部分、W4A16 + GRQ、RK3576 NPU**；视觉编码器尚未纳入这个基线。

## 1. 已验证成果

```text
Qwen/Qwen3.5-2B（Hugging Face）
  ↓ RKLLM-Toolkit 1.3.0 / Ubuntu x86_64
  ↓ W4A16 + GRQ / target=rk3576 / 2 NPU cores
qwen3.5-2b-w4a16-grq-rk3576.rkllm（约 2.2GB）
  ↓ SHA-256 校验与传输
RK3576 / Ubuntu aarch64 / driver 0.9.7
  ↓ RKLLM Runtime 1.3.0
中文文本正常生成
```

当前实测基线：

| 项目 | 结果 |
|---|---:|
| Prompt tokens | 20 |
| 实际生成 tokens | 55 |
| 三次平均 Prefill | 26.37 tok/s |
| 三次平均 Decode | 6.14 tok/s |
| 一次 Runtime 日志 Peak Memory | 1501 MB |

这些数字只描述本次条件，不能脱离版本、Prompt、上下文、频率与测量口径横向比较。

## 2. 环境冻结

### 2.1 Host：模型转换端

| 项目 | 已验证配置 |
|---|---|
| OS | Ubuntu 22.04.5 LTS x86_64 |
| CPU/RAM | 4 核 / 8GB |
| Python | 3.10.12 venv |
| Swap | 原 2GB + 16GB 临时 swap |
| Toolkit | rkllm-toolkit 1.3.0 cp310 x86_64 |
| 磁盘 | 120GB 虚拟盘，转换前保证数十 GB 余量 |

### 2.2 Board：推理端

| 项目 | 已验证配置 |
|---|---|
| SoC | RK3576 |
| OS | KickPi Ubuntu 24.04.3 LTS aarch64 |
| RAM/存储 | 4GB / 32GB（系统识别约 29GB） |
| NPU Driver | 0.9.7 |
| Runtime | rknn-llm release-v1.3.0 的 `librkllmrt.so` |
| 首次运行 | `max_new_tokens=128`，`max_context_len=2048` |

### 2.3 开始前生成 manifest

至少保存：

```bash
python --version
python -m pip show rkllm-toolkit
git -C ~/edge_llm/src/rknn-llm rev-parse HEAD
sha256sum ~/edge_llm/output/*.rkllm
```

板端保存：

```bash
uname -a
cat /etc/os-release
sudo cat /sys/kernel/debug/rknpu/version
sha256sum ./lib/librkllmrt.so
sha256sum ~/edge_llm/models/*.rkllm
```

若驱动节点在当前 BSP 中不同，应把实际节点写入 manifest，而不是为了匹配教程去修改系统。

## 3. Host 准备

### 3.1 磁盘和 swap

```bash
lsblk
df -h /
free -h
swapon --show
```

8GB RAM 的虚拟机可以完成这次转换，但量化阶段可能需要更多内存。创建 16GB swap：

```bash
sudo fallocate -l 16G /swapfile-rkllm
sudo chmod 600 /swapfile-rkllm
sudo mkswap /swapfile-rkllm
sudo swapon /swapfile-rkllm
```

若确实需要开机自动挂载，再确认 `/etc/fstab` 中没有重复项后添加。Swap 只避免 OOM，CPU 量化进入频繁换页时会显著变慢。

### 3.2 Python 3.10 独立环境

```bash
sudo apt update
sudo apt install -y \
  python3.10-venv python3.10-dev \
  git git-lfs wget curl unzip \
  build-essential cmake pkg-config

git lfs install
mkdir -p ~/edge_llm/{models,output,src,logs,manifests}

/usr/bin/python3.10 -m venv ~/edge_llm/.venv-rkllm
source ~/edge_llm/.venv-rkllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

检查点：

```bash
python --version
which python
```

不修改系统 `/usr/bin/python3`，也不把不受 wheel 支持的 Python 版本硬塞进流程。

## 4. 固定 RKLLM 版本

```bash
source ~/edge_llm/.venv-rkllm/bin/activate
cd ~/edge_llm/src

git clone --branch release-v1.3.0 --depth 1 \
  https://github.com/airockchip/rknn-llm.git

cd rknn-llm
git rev-parse HEAD

python -m pip install \
  ./rkllm-toolkit/packages/rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl
```

验证：

```bash
python - <<'PY'
from rkllm.api import RKLLM
print("RKLLM Toolkit import OK")
PY
```

Qwen3.5 架构支持来自较新的 RKLLM 版本。本实战只声明 1.3.0 已实际跑通，不把不同版本混用当作可支持方案。

## 5. 下载并校验模型

```bash
source ~/edge_llm/.venv-rkllm/bin/activate
python -m pip install -U huggingface_hub

cd ~/edge_llm/models
hf download Qwen/Qwen3.5-2B --local-dir Qwen3.5-2B
```

检查：

```bash
du -sh ~/edge_llm/models/Qwen3.5-2B
find ~/edge_llm/models/Qwen3.5-2B -maxdepth 1 -type f -printf '%f %s\n' | sort
```

本次原始目录约 4.57GB，主权重约 4.55GB。若看到很小的文本指针而非大权重文件，通常是 Git LFS/下载未完成。

固定模型 revision：

```bash
git -C ~/edge_llm/models/Qwen3.5-2B rev-parse HEAD 2>/dev/null || true
```

使用 `hf download` 时也可在 manifest 中记录下载时的 commit hash，保证 Processor、Tokenizer 与权重一致。

## 6. 生成量化校准输入

本次使用 RKLLM 1.3.0 Qwen3.5 多模态示例生成语言模型 `inputs_embeds` 校准数据：

```bash
source ~/edge_llm/.venv-rkllm/bin/activate
cd ~/edge_llm/src/rknn-llm/examples/multimodal_model_demo

python data/make_input_embeds_for_quantize.py \
  --path ~/edge_llm/models/Qwen3.5-2B \
  --model_type qwen3.5
```

本次结果为 20 个样本，并生成：

```text
data/llm_inputs.json
data/llm_inputs/*
```

本版本示例中生成器与导出脚本使用的文件名曾出现不一致，因此先检查，再按实际结构兼容：

```bash
test -e data/inputs.json || ln -s llm_inputs.json data/inputs.json
ls -l data/*inputs.json
```

这属于**已观察到的特定版本样例问题**。升级版本后先检查源码，不要无条件创建链接。

校准集的意义和改进方法见 [03_RKLLM转换_量化与W4A16_GRQ](./03_RKLLM转换_量化与W4A16_GRQ.md)。20 个示例可以跑通工具链，不等于已经覆盖你的中文、代码、视觉和行业数据分布。

## 7. 导出 W4A16 + GRQ 制品

在示例目录新建自己的脚本，不修改官方文件。核心代码：

```python
from pathlib import Path
from rkllm.api import RKLLM

model_path = "/home/<user>/edge_llm/models/Qwen3.5-2B"
dataset_path = "data/inputs.json"
output_path = (
    "/home/<user>/edge_llm/output/"
    "qwen3.5-2b-w4a16-grq-rk3576.rkllm"
)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
llm = RKLLM()

ret = llm.load_huggingface(
    model=model_path,
    device="cpu",
    dtype="bfloat16",
)
if ret != 0:
    raise RuntimeError(f"load_huggingface failed: {ret}")

ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w4a16",
    quantized_algorithm="grq",
    target_platform="rk3576",
    num_npu_core=2,
    dataset=dataset_path,
    extra_qparams=None,
    hybrid_rate=0,
    max_context=4096,
)
if ret != 0:
    raise RuntimeError(f"build failed: {ret}")

ret = llm.export_rkllm(output_path)
if ret != 0:
    raise RuntimeError(f"export_rkllm failed: {ret}")
```

运行并保存完整日志：

```bash
source ~/edge_llm/.venv-rkllm/bin/activate
python convert_qwen35_2b_rk3576.py 2>&1 | \
  tee ~/edge_llm/logs/export_w4a16_grq.log
```

另开终端观察资源：

```bash
watch -n 2 'free -h; df -h /'
```

### 7.1 关键参数的工程含义

| 参数 | 当前值 | 作用 |
|---|---|---|
| `do_quantization` | true | 执行低比特量化 |
| `quantized_dtype` | w4a16 | 主要权重 4 bit、激活 16 bit 路线 |
| `quantized_algorithm` | grq | Rockchip 工具链支持的量化算法选择 |
| `target_platform` | rk3576 | 生成目标平台制品 |
| `num_npu_core` | 2 | 构建时指定核心策略；需做实测对比 |
| `max_context` | 4096 | 构建上限，不等于每次运行都必须用 4096 |
| `hybrid_rate` | 0 | 当前纯 NPU 路线的构建参数，语义以版本文档为准 |

不要把 `grq` 的内部算法细节凭名称自行推导；除非官方材料明确说明，否则只陈述接口、校准需求和实测效果。

### 7.2 已观察到的非致命提示

- CPU 不支持 BF16 时 Toolkit 可能回退 FP32；
- Fast path 不可用时可能回退普通 PyTorch；
- 日志提示只导出 `Qwen3_5ForCausalLM`，符合当前“纯文本第一阶段”；
- CPU GRQ 优化可能长时间无输出，本次累计约 2 小时。

是否成功由 API 返回值、成功终态、输出文件和后续板端初始化共同判断。

导出后立即生成哈希：

```bash
sha256sum ~/edge_llm/output/qwen3.5-2b-w4a16-grq-rk3576.rkllm | \
  tee ~/edge_llm/manifests/model.sha256
ls -lh ~/edge_llm/output/*.rkllm
```

## 8. 板端准备与传输

### 8.1 硬件与驱动检查

```bash
uname -m
cat /etc/os-release
free -h
df -h /
sudo cat /sys/kernel/debug/rknpu/version
dmesg | grep -i rknpu | tail -30
```

当前有效结果包括 `aarch64` 和 `RKNPU driver: v0.9.7`。某些 BSP 不暴露 `/dev/rknpu*`；不能只据此判定 NPU 不可用，最终还要看驱动日志与 Runtime 初始化。

### 8.2 OOM 保险

4GB 板端可配置 4GB swap 作为保护：

```bash
sudo fallocate -l 4G /swapfile-rkllm
sudo chmod 600 /swapfile-rkllm
sudo mkswap /swapfile-rkllm
sudo swapon /swapfile-rkllm
```

它不提升推理速度。若实际交互发生持续换页，应按 [06_4GB内存容量工程与测量方法](./06_4GB内存容量工程与测量方法.md) 缩减并发组件、上下文或缓冲区。

### 8.3 传输与校验

从 Host 执行：

```bash
scp ~/edge_llm/output/qwen3.5-2b-w4a16-grq-rk3576.rkllm \
  <board-user>@<board-ip>:/home/<board-user>/edge_llm/models/

scp -r ~/edge_llm/src/rknn-llm \
  <board-user>@<board-ip>:/home/<board-user>/edge_llm/src/
```

分别在 Host 和板端执行 `sha256sum`，结果必须完全一致。

## 9. 板端原生编译官方 Demo

本机已有 CMake/G++ 时：

```bash
cd ~/edge_llm/src/rknn-llm/examples/rkllm_api_demo/deploy
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j4
cmake --install build-native
```

检查产物：

```text
install/demo_Linux_aarch64/
├─ llm_demo
└─ lib/librkllmrt.so
```

在运行前：

```bash
cd install/demo_Linux_aarch64
file ./llm_demo ./lib/librkllmrt.so
ldd ./llm_demo
```

官方 Demo 是最小金标准。自研 Qt/C++ 封装失败时，先回到这个 Demo 判断是模型/Runtime 还是应用代码。

## 10. 首次运行

```bash
cd ~/edge_llm/src/rknn-llm/examples/rkllm_api_demo/deploy/install/demo_Linux_aarch64

export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
export RKLLM_LOG_LEVEL=1

./llm_demo \
  /home/<board-user>/edge_llm/models/qwen3.5-2b-w4a16-grq-rk3576.rkllm \
  128 \
  2048
```

参数顺序为：模型路径、`max_new_tokens`、`max_context_len`。首次用 2048，是为了给 4GB 板卡留余量；导出上限 4096 并不要求运行时立刻使用 4096。

输入：

```text
请用三句话解释什么是 NPU。
```

官方 Demo 默认 `keep_history=0`，所以可以连续输入，但每次是单轮问答。多轮对话必须自己定义结构化历史、Chat Template、裁剪和 Prompt Cache。

## 11. 固定基准与结果

命令：

```bash
mkdir -p ~/edge_llm/logs

printf '请用三句话解释什么是NPU。\nexit\n' | \
./llm_demo \
  /home/<board-user>/edge_llm/models/qwen3.5-2b-w4a16-grq-rk3576.rkllm \
  64 \
  2048 | tee ~/edge_llm/logs/w4a16_baseline_1.log
```

连续运行三次的结果：

| 运行 | Prefill tokens | Prefill ms | Prefill tok/s | Generate tokens | Generate ms | Decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 785.74 | 25.45 | 55 | 9114.95 | 6.03 |
| 2 | 20 | 811.55 | 24.64 | 55 | 8877.66 | 6.20 |
| 3 | 20 | 689.52 | 29.01 | 55 | 8872.76 | 6.20 |
| 平均 | 20 | 762.27 | 26.37 | 55 | 8955.12 | 6.14 |

一次日志记录 `Peak Memory Usage: 1501 MB`。下一轮应升级为预热后至少 10 次、报告 p50/p95，并用 PSS/CMA/温度采样补全口径，见 [05_RK3576性能模型_GEMM_GEMV与Roofline](./05_RK3576性能模型_GEMM_GEMV与Roofline.md)。

## 12. 关键故障与已验证处理

### 12.1 Hugging Face 网络不可达

VMware NAT 中的 `127.0.0.1` 是虚拟机自身。若代理在 Windows，需要开启局域网访问，并使用 VMnet8 宿主机地址与代理端口。代理配置属于当前网络环境，不应写死到通用脚本，更不能把密钥提交进仓库。

### 12.2 量化阶段长时间没有新日志

不要立即中断。另开终端查看进程 CPU、运行时间、内存与磁盘。只有进程退出、内核 OOM、磁盘写满或工具明确报错，才判断失败。

### 12.3 `data/inputs.json` 不存在

先确认当前 release 的生成器与导出器实际文件名；1.3.0 本次实战用符号链接兼容。升级后重新核实。

### 12.4 `rkllm init failed`

按优先级检查：

1. Host/Board 制品哈希；
2. RK3576 target；
3. Toolkit、Runtime、Header 版本；
4. 实际加载的动态库路径；
5. 驱动；
6. 可用内存/CMA；
7. 官方 Demo 是否同样失败。

详见 [09_版本矩阵_故障定位与回归测试](./09_版本矩阵_故障定位与回归测试.md)。

## 13. 从“跑通”到“本地设备助手”

按以下顺序演进，每一步都保留回归基线：

1. C++ RAII 封装 RKLLM handle、Callback 和 Abort；
2. 单模型串行请求队列；
3. Qt 流式聊天界面，不阻塞 GUI；
4. 本地 `/v1/chat/completions` + SSE，形成类似 Ollama 的服务边界；
5. 结构化历史、上下文裁剪和 Prompt Cache；
6. IMX415 稳定预览；
7. Qwen3.5 视觉编码器导出为 RKNN；
8. Vision embedding 与 RKLLM 多模态输入闭环；
9. 全功能 4GB 容量、温度、稳定性测试；
10. 再接语音、RAG、传感器和远程管理。

这一顺序让纯文本、服务层、相机、视觉模型和跨模态桥接可以分别定位，避免一次集成五个未知量。

## 14. 本章最终验收

- 新机器按文档可重新生成 `.rkllm`；
- 制品和动态库都有 SHA-256；
- 官方 Demo 在 RK3576 上稳定生成中文；
- 基准能复现到合理误差范围并记录完整条件；
- 故障可回落到明确的阶段；
- 视觉未完成的部分被明确标记，不把纯文本闭环包装成完整 VLM；
- 下一步 Qt 服务化与 IMX415 多模态均有清晰接口。

## 官方参考

- [Rockchip RKLLM](https://github.com/airockchip/rknn-llm)
- [RKLLM release-v1.3.0](https://github.com/airockchip/rknn-llm/releases/tag/release-v1.3.0)
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [RKLLM Multimodal Demo](https://github.com/airockchip/rknn-llm/tree/release-v1.3.0/examples/multimodal_model_demo)

## 原始实战记录

更偏“当天操作日志”的原始版本未纳入本仓库；本章负责建立可复现、可验证、可扩展的正式版本。

