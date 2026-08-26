---
tags:
  - RKLLM
  - C++
  - Qt
  - OpenAI-API
aliases:
  - RKLLM服务化
---

# RKLLM C API：Runtime 封装与服务化

> `llm_demo` 证明模型能跑；产品化则要解决生命周期、线程、流式回调、中止、会话、错误恢复和 API 边界。本章把 RKLLM 当作一个需要严谨封装的本地推理服务。

## 1. 先定义产品边界

推荐把系统拆为两层：

```text
Qt 触屏客户端 / 浏览器 / 局域网设备
                │ HTTP + SSE，或进程内接口
                ▼
       RKLLM Service（唯一模型所有者）
       ├─ 请求队列与状态机
       ├─ 会话/模板/采样
       ├─ Callback → token event
       ├─ 指标、日志与健康检查
       └─ RKLLM Runtime / NPU Driver
```

4GB 板卡上不建议为了“前后端分离”启动多份模型进程。无论 Qt 是进程内调用还是访问本机 HTTP 服务，都应只有一个明确的 Runtime 所有者。

## 2. 官方 C API 生命周期

根据 RKLLM 1.3.0 头文件，核心调用链为：

```text
rkllm_createDefaultParam
        ↓ 修改模型路径、上下文等参数
rkllm_init(handle, param, callback)
        ↓
rkllm_run / rkllm_run_async
        ↓ callback(NORMAL / WAITING / FINISH / ERROR)
rkllm_abort（可选）
        ↓
rkllm_destroy
```

另有 `rkllm_is_running`、Prompt Cache 保存/加载等接口。实际函数签名必须以**与当前 Runtime 同版本的 `rkllm.h`**为准，不能拿 1.2.x 头文件链接 1.3.0 动态库。

## 3. C++ 封装的状态机

建议显式定义：

```text
Uninitialized → Loading → Ready → Running → Ready
                    │         │        │
                    └─────────┴────────┴→ Error
Ready/Running → ShuttingDown → Uninitialized
```

关键不变量：

- `handle == nullptr` 时不能运行；
- 同一 handle 是否允许并行请求，由当前 Runtime 能力和实测决定；默认按串行队列设计；
- `Running` 时切模型，必须先中止并等待回调结束；
- Callback 不直接更新 Qt 控件；
- 析构前必须保证 Runtime 不再回调已释放对象。

## 4. 一个推荐的 RAII 轮廓

以下是结构示意，不替代当前 SDK 头文件：

```cpp
class RkllmEngine {
public:
    using TokenSink = std::function<void(TokenEvent)>;

    Result load(const EngineConfig& config);
    Result submit(const Request& request, TokenSink sink);
    Result abort();
    EngineState state() const;
    Metrics lastMetrics() const;
    ~RkllmEngine();

private:
    static int callbackThunk(/* use exact rkllm.h signature */);
    int onCallback(/* result, state, userdata... */);

    RKLLM_Handle handle_{};
    std::mutex mutex_;
    EngineState state_{EngineState::Uninitialized};
    std::shared_ptr<RequestContext> active_;
};
```

`RequestContext` 至少持有：请求 ID、取消标志、累积文本、开始时间、首 token 时间、采样参数、错误码和安全的事件队列。

## 5. Callback 不能做重活

官方 Runtime 通过 Callback 送出结果和状态。回调线程的原则：

1. 复制当前事件所需的最小数据；
2. 写入有界队列；
3. 立即返回；
4. 由消费线程完成 JSON、网络发送、Markdown 渲染和 Qt 更新。

如果在 Callback 中直接操作 UI、做磁盘日志或阻塞网络，可能造成 token 抖动，甚至与中止/销毁形成死锁。

Callback 状态需要转成应用事件：

| Runtime 状态 | 应用事件 | 行为 |
|---|---|---|
| NORMAL | `delta` | 追加 token/文本片段 |
| WAITING | `waiting` | 记录状态，不能误判结束 |
| FINISH | `done` | 封口指标、释放当前请求 |
| ERROR | `error` | 保存错误、恢复或进入 Error |

Callback 的返回值语义也以对应版本头文件为准；不要凭经验随意返回非零值。

## 6. 同步、异步与取消

### 6.1 选择策略

- 最小 Demo：同步调用可以减少状态复杂度；
- Qt 产品：推理不能阻塞 UI 线程，使用工作线程或异步接口；
- HTTP 服务：请求线程负责入队，独立推理线程拥有 handle；
- 中止：UI 的“停止生成”映射到 `rkllm_abort`，并等待终态事件后再接受下一请求。

取消不是“客户端断开就丢掉指针”。必须定义：

```text
disconnect / stop button
        ↓ set cancelled
        ↓ rkllm_abort
        ↓ wait FINISH/ERROR or bounded timeout
        ↓ close SSE + release RequestContext
```

## 7. 会话历史与 Prompt Cache

官方 Demo 的 `keep_history=0` 表示连续输入仍是单轮问答。产品需要自己决定：

- 历史由应用拼成 Chat Template，还是交给 Runtime 的历史机制；
- 历史 token 超限时如何裁剪；
- System Prompt 是否固定；
- Prompt Cache 与模型、模板、版本是否绑定；
- Cache 失效后如何回退到普通 Prefill。

建议应用层保存结构化消息，发送前再渲染模板。不要只保存已经拼接的长字符串，否则难以裁剪、审计和切换模板。

## 8. 本地 OpenAI-compatible API

RKLLM 官方仓库提供 Server Demo，并支持典型接口：

```text
GET  /v1/models
POST /v1/chat/completions
```

产品可另外提供：

```text
GET  /health
GET  /metrics
POST /v1/chat/completions
POST /internal/abort/{request_id}
```

流式响应使用 SSE：

```text
data: {"id":"...","choices":[{"delta":{"content":"你"}}]}
data: {"id":"...","choices":[{"delta":{"content":"好"}}]}
data: [DONE]
```

“OpenAI-compatible”不代表所有字段完全一致。必须列出支持矩阵：

| 能力 | 状态 | 说明 |
|---|---|---|
| messages | 支持 | 应用渲染 Chat Template |
| stream | 支持 | SSE |
| temperature/top_p | 待逐项验证 | 映射到每请求采样参数 |
| tools | 版本相关 | 官方 Server Demo 有相关路径，需用目标模型实测 |
| usage token 数 | 待验证 | 需要可信 tokenizer 计数 |
| parallel requests | 默认不承诺 | 单 handle 串行队列 |

## 9. Qt 客户端的两种落地方式

### 9.1 进程内封装

优点：少一层序列化，部署简单。缺点：Runtime 崩溃会带走 UI，模型生命周期与界面耦合。

线程结构：

```text
Qt GUI Thread
   │ signal/slot
Inference Worker Thread
   │ bounded event queue
RKLLM Callback
```

### 9.2 本机独立服务

优点：Qt、浏览器和局域网都可使用；推理崩溃可单独拉起；更接近 Ollama 的产品形态。缺点：增加服务进程、HTTP/SSE 和运维成本。

对 4GB 设备，这一成本主要在服务框架和复制，而不是再加载一份模型——前提是 Qt 只作为客户端。可先用轻量 C++ HTTP 库或验证官方 Flask Demo，再依据实测决定是否替换；不能预先认定 Python 服务一定不可用。

## 10. 守护、日志与安全

- 使用 systemd 启动服务，设置合理的失败重启；
- 健康检查要区分“进程活着”“模型已加载”“当前忙碌”；
- 日志记录 request_id、版本、耗时和错误，不默认记录隐私提示词；
- 限制请求体、上下文和生成长度，防止内存被恶意占满；
- 默认只监听 `127.0.0.1`；开放局域网时增加鉴权与防火墙；
- 模型路径、动态库路径和配置使用绝对路径，避免 systemd 工作目录差异；
- Runtime 发生不可恢复错误时，让服务进程有序退出，由 systemd 拉起，比在损坏状态上无限重试更可靠。

## 11. 测试清单

1. 初始化失败是否返回明确错误；
2. 正常流式生成是否只发一次 `[DONE]`；
3. 用户连续点击两次发送是否排队而不是并发踩 handle；
4. 中止后能否立即发起下一请求；
5. 客户端断开时推理是否按策略中止；
6. 运行中退出 Qt 是否安全销毁；
7. 超长历史是否在发送前被拒绝或裁剪；
8. 100 轮后句柄、线程和内存是否稳定；
9. Runtime 崩溃后服务能否恢复；
10. API 输出与官方 Server Demo 的兼容字段是否经过自动测试。

## 12. 验收标准

- RKLLM handle 只有一个清晰所有者；
- UI、网络与 Callback 线程不存在交叉阻塞；
- 支持流式输出、中止、错误终态和有界队列；
- OpenAI-compatible 有明确的字段支持矩阵；
- 版本不匹配能在启动阶段被发现；
- Qt 进程内与独立服务两种模式的取舍有实测依据。

## 官方参考

- [RKLLM C API 头文件](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)
- [RKLLM C++ Demo](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp)
- [RKLLM Server Demo](https://github.com/airockchip/rknn-llm/tree/main/examples/rkllm_server_demo)

