# Frigate Vision

[![Tests](https://github.com/kaattz/ha-frigate-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/kaattz/ha-frigate-vision/actions/workflows/tests.yml)
[![HACS](https://github.com/kaattz/ha-frigate-vision/actions/workflows/hacs.yml/badge.svg)](https://github.com/kaattz/ha-frigate-vision/actions/workflows/hacs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Home Assistant 自定义集成（HACS）：把**任意受监控区域的活动**编排成一条可复用、可验证、可回滚的流水线——用事件固定活动边界，用 Frigate 检测与 Review 提供视觉证据，合成帧联系图后交给兼容 OpenAI 协议的视觉模型分析，最后通过通知蓝图交付结果。

集成本身**与场景无关**：门口只是它的参考场景——核心的状态机、抽帧、联系图、幂等与交付并不知道「门」是什么。换一个摄像头和一套语义，它就是车道、后院、走廊或库房的 activity intelligence。

> **状态：0.1.0，实验性。** 本地实现与单元测试已完成，但**现场 48 小时验收尚未通过**（见 [validation-checklist.md](docs/validation-checklist.md)）。当前只应部署在 `observe` 模式观察，不要直接用于生产通知。

## 它做什么

```text
事件源（可选，仅门锁场景）          Frigate events + reviews
门锁 / 门磁 / 门铃                        │
        └──────────────┬─────────────────┘
                       ↓
                 frigate_vision
   ┌───────────────────────────────────────────┐
   │ 通用核心（与场景无关）                      │
   │  ├─ 活动状态机与持久化                      │
   │  ├─ Review 归属与活动合并（detection ID）    │
   │  ├─ 严格幂等 + 重启恢复                     │
   │  ├─ 帧选择与联系图合成                      │
   │  ├─ observe / shadow / live                │
   │  └─ 兼容 OpenAI 协议的视觉模型调用           │
   ├───────────────────────────────────────────┤
   │ 场景层（scenes.py，可扩展）                 │
   │  └─ 每个场景声明：可回答的分类 + 提示词       │
   │     + 允许读取的辅助信号 + 提示词版本         │
   └───────────────────────────────────────────┘
                       ↓
               标准化活动结果事件
                       ↓
           通知蓝图 / 你自己的自动化
```

- **活动边界来自事件，不来自猜测。** 配了门锁就以开门建立周期、关门固定边界，重复开门只记录不覆盖；没配门锁则退化为 **review-only**，直接以 Frigate 人物 Review 为活动单元——后者对任意摄像头都成立。
- **证据是确定性抽帧。** 单向出入用 3 帧联系图；门未关往返用 2×3 联系图；通用 Review 用「首帧 + 3 张高变化帧 + 末帧 + 后置现场帧」。优先采用 Frigate `path_data` 做路径运动选帧，缺失时回退像素差分。
- **严格幂等。** 媒体生成、模型调用、通知交付都先持久化 `started` 阶段再执行；结果不确定的失败（`analysis_outcome_unknown`、`delivery_outcome_unknown`）**禁止自动重试**，宁可人工介入也不重复打扰或重复计费。
- **模式可回滚。** `observe` → `shadow` → `live` 逐级放开，任何一步都能退回上一级而不丢历史。

## 能力一览

### 场景层：泛化的关键

分类并不是硬编码在流程里的，而是由 `scenes.py` 中的**场景**声明。一个场景定义四件事：

| 场景声明 | 作用 |
|---|---|
| `classifications` | 该场景**允许**返回的分类集合；越界值在本地被拒绝 |
| `template` | 提示词：这几帧画面意味着什么，以及**不可以**推断什么 |
| `signals` | 允许读取的辅助信号（如门锁内外侧）；未声明的信号**根本不会进入**提示词 |
| `prompt_version` | 提示词版本，参与分析缓存键；措辞变了就不会复用旧结果 |

信号隔离是刻意的：门口场景可以读门锁状态，通用 Review 场景**不能**——它拿不到任何 zone telemetry（实测 90 条独立 Review 中 0 条携带 `detection_zone_updates`，而门周期有），所以一个暗示「知道门的状态」的通用提示词，等于要求模型编造证据。

**新增一个场景 = 在 `scenes.py` 注册一个 `Scene`**，核心的抽帧、联系图、调用、交付、幂等全部无需改动，且新场景不会意外依赖另一个场景的上下文。测试 [test_scenes.py](tests/test_scenes.py) 就以「以后加车道、后院场景」为前提验证这套接口。

当前内置场景：

| 场景 | 用途 | 可返回分类 |
|---|---|---|
| `review_six` | 任意摄像头的人物 Review（通用） | 全部 12 类 |
| `door_single` | 门锁单向出入（3 帧） | `home_arrival`、`home_departure`、`unknown_activity`、`unable_to_confirm` |
| `door_roundtrip` | 门未关短时往返（2×3 帧） | `short_roundtrip`、`unknown_activity`、`unable_to_confirm` |

可返回分类全集：`home_arrival`、`home_departure`、`short_roundtrip`、`package_delivery`、`food_delivery`、`cleaning`、`maintenance`、`visitor`、`elevator_activity`、`suspicious_activity`、`unknown_activity`、`unable_to_confirm`。

> 需要提醒的是：新增场景要自己负责提示词质量与验证。`door_*` 场景的措辞是在真实现场数据上反复测量调优的结果（例如「不能凭电梯方向断言进入可见电梯」），新场景应比照 [validation-checklist.md](docs/validation-checklist.md) 自行验收。

### 实体

| 实体 | 说明 |
|---|---|
| `select.<name>_processing_mode` | 运行模式，切换后自动 reload |
| `event.<name>_activity` | 活动完成 / 失败事件 |
| `sensor.<name>_last_classification` | 最近一次分类 |
| `sensor.<name>_last_confidence` | 最近一次置信度 |
| `sensor.<name>_last_error` | 最近一次稳定错误码 |
| `sensor.<name>_pending_activities` | 未结束的活动数 |
| `binary_sensor.<name>_healthy` | 运行时健康状态 |

**服务**：

| 服务 | 用途 |
|---|---|
| `frigate_vision.get_activity` | 返回脱敏后的活动摘要（带响应数据） |
| `frigate_vision.process_review` | 手动把某条未处理的 person Review 入队 |
| `frigate_vision.retry_failed` | 仅对**可证明安全**的失败建立一次显式重试 |
| `frigate_vision.ack_delivery` | 通知蓝图处理成功后回执，活动转为 `completed` |

**交付事件** `frigate_vision_activity` 字段：`entry_id`、`activity_id`、`delivery_attempt_id`、`classification`、`description`、`confidence`、`evidence_url`、`evidence_image_url`、`evidence_offsets`、`clip_url`、`hls_url`、`frigate_review_url`、`review_ids`、`occurred_at`。

其中：

- `evidence_url` 是 `media-source://` 标识，供 Home Assistant 媒体浏览器使用。
- `evidence_image_url` 是**已签名**的相对 HTTP 地址，供回放弹窗内的六宫格显示。`<img>` 带不上 `Authorization` 头，HA 的鉴权中间件也没有 cookie 通道，因此未签名的 `/api/` 图片必然 401；签名与路径精确绑定，也不能复用视频的签名。
- `evidence_offsets` 是六宫格每格对应的**视频秒数**（相对片段起点，已夹取到窗口内），**竖线**分隔，如 `5.3|11.2|48.1`。分隔符不能是逗号：HA 的原生模板解析器会把逗号分隔的模板结果当作 tuple，导致通知脚本以 `TypeError: TupleWrapper is not JSON serializable` 失败，整条通知都写不进去。

## 安装

需要 Home Assistant **2026.8.0+**、MQTT 集成、以及已接入 HA 的 Frigate。

### HACS（推荐）

1. HACS → 集成 → 右上角菜单 → 自定义存储库，添加 `https://github.com/kaattz/ha-frigate-vision`，类别选 **Integration**。
2. 安装后重启 Home Assistant。
3. 设置 → 设备与服务 → 添加集成 → **Frigate Vision**。

### 手动安装

把 `custom_components/frigate_vision/` 复制到 HA 配置目录的 `custom_components/` 下并重启。

### 配置项

**每个监控点添加一个配置项**（一个门口、一路摄像头、一个区域），同一台 HA 可以并存多个，彼此不共享可变状态。

- **Frigate**：base URL、认证方式（`none` / `native`）、MQTT topic 前缀（默认 `frigate`）、精确的摄像头名。
  - 端口 5000 是无认证内部 API，只应在可信内网使用；端口 8971 使用原生认证，填写用户名/密码后由共享 CookieJar 登录并自动刷新。
- **门口设备（可选）**：门锁 `event` 实体、动作与内外侧属性名及其匹配值。**留空门锁实体即为 review-only 模式**，只用 Frigate 人物 Review，不建立门周期；此时门磁与门铃字段会被拒绝（`door_lock_required`）。非门口场景请直接留空，切到 review-only。
- **视觉 Provider**：base URL（默认 `https://api.deepseek.com/v1`）、API Key、模型名、推理档位（`default` / `low` / `high` / `max`）。保存前可点「测试连接」实测一次往返。
- **行为选项**：处理模式、图片宽度、最大 Token、输出语言、最短活动时长、队列上限、历史与媒体保留天数。

### 运行模式

| 模式 | 行为 | 外部副作用 |
|---|---|---|
| `observe` | 只做关联与证据，跑通到 `evidence_ready` | 0 次模型调用、0 次通知 |
| `shadow` | 证据 + 一次模型调用，保存结果 | 有模型调用，**不发**正式结果事件 |
| `live` | 证据 + 模型调用 + 一次需回执的交付 | 模型调用 + 通知各一次 |

迁移到 `shadow` / `live` 前，请按 [migration.md](docs/migration.md) 先停用（**不要删除**）被替代的旧自动化，以便随时回滚。

## 通知蓝图

导入地址：

```text
https://github.com/kaattz/ha-frigate-vision/blob/main/blueprints/automation/frigate_vision/activity_notification.yaml
```

蓝图监听 `frigate_vision_activity` 事件，负责分类过滤、静音时段、标题正文格式、证据图与录像链接，并在动作执行后调用 `ack_delivery`。集成本身**不硬编码任何 notify 服务**。

蓝图内附带的录像链接走 HA 自身的鉴权代理（`/api/frigate_vision/clip/<entry_id>/<activity_id>.mp4`），因此外网也能打开，且不会把 Frigate 暴露到公网。

## 隐私与数据

- 联系图保存在 HA 媒体目录的 `frigate_vision/<entry_id>/<activity_id>.jpg`，通过 `media-source://frigate_vision/...` 或需登录的 `/api/frigate_vision/media/...` 端点访问，**不会**作为静态文件公开。
- 联系图会上传给**你自己配置的**视觉 Provider。请确认该 Provider 的隐私条款可接受。
- 诊断信息（`diagnostics.py`）已脱敏：不含 API Key、图片、本地路径、原始 MQTT 报文和完整模型回复。
- 仓库 `.gitignore` 明确排除 `evidence-review/` 与 `.e2e-media/`——真实画面证据**永远不要提交**。

## 故障排查

集成所有失败都带稳定错误码，常见的几个：

| 错误码 | 含义与处理 |
|---|---|
| `analysis_outcome_unknown` | 模型可能已收到请求，**禁止自动重试** |
| `delivery_outcome_unknown` | 通知可能已发出，**不要补发** |
| `media_retry_exhausted` | 可安全使用 `retry_failed` |
| `ambiguous_review_ownership` | Review 归属不唯一，检查 detection ID 与 zone 布局 |
| `invalid_path_data` | Frigate 路径数据非法，永久失败，不回退像素差分 |
| `door_open_too_long` | 只收到开门未收到关门，30 分钟后看门狗将周期置为失败（绝不臆造关门时间）；仅在使用门锁场景时出现 |

其余错误码、Repair 说明与处理方法见 [troubleshooting.md](docs/troubleshooting.md)。可从配置项下载诊断信息定位问题。

## 兼容性

| 组件 | 支持基线 |
|---|---|
| Home Assistant | 2026.8.0+（实测 2026.8.3） |
| Frigate | 0.17.2 公开 HTTP / MQTT 契约 |
| Python | 3.14.2+ |
| 视觉 Provider | 兼容 OpenAI Chat Completions 的接口 |

集成只使用 Frigate 的公开 HTTP API 与 MQTT topic、HA 的公开扩展 API（`Store`、`mqtt.client.async_subscribe`、`media_source`、`issue_registry`），不 import 任何上游集成的内部模块。详见 [compatibility.md](docs/compatibility.md)。

## 开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install homeassistant==2026.8.3 pytest pytest-asyncio pytest-homeassistant-custom-component ruff mypy pillow pyyaml

python -m pytest -q                          # 26 个测试模块
ruff check custom_components tests
mypy custom_components/frigate_vision        # strict 模式
```

> Home Assistant 依赖 Unix 的 `fcntl`，**在 Windows 上无法直接跑 HA 相关 pytest**。Windows 本地只做 ruff / mypy / `compileall`，完整测试请在 WSL、Linux 或 CI 中运行。

`tests/e2e_llm_live.py` 是需要真实网络与凭据的**选择性**实弹测试，不在默认收集中。

## 非目标

- 不实现或复制任何 Provider 客户端生态，不替代 Frigate 的检测、Review、录像保留与 zone 编辑器。
- 不做人脸识别、身份推断或跨摄像头追踪。
- 不按固定时长猜测「回家 / 离家 / 倒垃圾 / 保洁」。
- 第一版不使用对象存储，不回退到完整视频 LLM。
- 不自动删除既有 HA 自动化、Helpers 或历史数据。
- 不内置除门口参考场景之外的场景实现；新增场景由使用者自行撰写并验收（见[场景层](#场景层泛化的关键)）。

## 文档

- [配置详解](docs/configuration.md)
- [迁移与回滚](docs/migration.md)
- [故障排查](docs/troubleshooting.md)
- [现场验收清单](docs/validation-checklist.md)
- [实施状态与漂移检查](docs/implementation-status.md)
- [兼容性与公开接口证据](docs/compatibility.md)
- 设计、需求、规格与计划：[docs/plans/](docs/plans/)、[specs/](specs/)

## License

[MIT](LICENSE)
