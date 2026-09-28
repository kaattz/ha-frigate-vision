# Frigate Vision

[![Tests](https://github.com/kaattz/ha-frigate-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/kaattz/ha-frigate-vision/actions/workflows/tests.yml)
[![HACS](https://github.com/kaattz/ha-frigate-vision/actions/workflows/hacs.yml/badge.svg)](https://github.com/kaattz/ha-frigate-vision/actions/workflows/hacs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Home Assistant 自定义集成（HACS）：把 Frigate 检测到的人物活动编排成一条**可复用、可验证、可回滚**的流水线——以 Frigate 人物 Review 固定活动边界，确定性抽帧合成联系图，交给兼容 OpenAI 协议的视觉模型分析，最后通过通知蓝图交付结果。

> **状态：0.1.0，实验性。** 实现与单元测试已完成，但**尚未有完整的现场验收记录**。配置完成即开始调用模型并推送通知，没有任何"只观察"的档位——请确认好 Provider 与蓝图后再启用。

## 它做什么

```text
                     Frigate events + reviews
                              │
                       frigate_vision
    ┌────────────────────────────────────────────┐
    │ 活动状态机与持久化                           │
    │ Review 归属与活动合并（detection ID）        │
    │ 严格幂等 + 重启恢复                          │
    │ 帧选择与联系图合成（6 格，必要时 9 格）        │
    │ 人物特写栏（可选）                           │
    ├────────────────────────────────────────────┤
    │ 视觉调用（兼容 OpenAI 协议）                  │
    │  ├─ 第一组 Provider                        │
    │  └─ 第二组 Provider（可选，第一组失败才用）    │
    ├────────────────────────────────────────────┤
    │ 场景层（scenes.py，内置一个场景）             │
    │  └─ review_six：可回答的分类 + 提示词         │
    │     + 提示词版本 + 部署自己的现场描述         │
    └────────────────────────────────────────────┘
                              ↓
                      标准化活动结果事件
                              ↓
                  通知蓝图 / 你自己的自动化
```

- **活动边界来自 Frigate Review，不来自猜测。** 每条人物 Review 就是一次活动；以 detection ID 关联同一活动的多次上报，`min_review_seconds` 与 zone 配置决定什么值得分析。
- **证据是确定性抽帧。** 「首帧 + 3 张高变化帧 + 末帧 + 后置现场帧」共 6 格，当画面中没有足够的变化候选时再补 3 格探测帧、扩为 3×3。优先采用 Frigate `path_data` 做路径运动选帧，缺失时回退像素差分。
- **严格幂等。** 媒体生成、模型调用、通知交付都先持久化 `started` 阶段再执行；结果不确定的失败（`analysis_outcome_unknown`、`delivery_outcome_unknown`）**禁止自动重试**，宁可人工介入也不重复打扰或重复计费。
- **供应商可故障转移（可选）。** 配了第二组 provider 时，第一组拿不到答案不会丢掉这条活动：同一次幂等声明内换第二组重试，共用同一张联系图。见[第二组 Provider](#第二组-provider故障转移可选)。

## 两个层次：核心与场景

分类不是硬编码在流程里的，而是由 `scenes.py` 中的**场景**声明。一个场景定义：

| 场景声明 | 作用 |
|---|---|
| `classifications` | 该场景**允许**返回的分类集合；越界值在本地被拒绝 |
| `template` | 提示词：这几帧画面意味着什么，以及**不可以**推断什么 |
| `signals` | 允许读取的辅助信号；未声明的信号**根本不会进入**提示词 |
| `prompt_version` | 提示词版本，参与分析缓存键；措辞变了就不会复用旧结果 |
| `accepts_scene_description` | 是否接受部署自己的现场布局描述（见下） |

内置场景只有一个：

| 场景 | 用途 | 声明可返回分类 |
|---|---|---|
| `review_six` | 摄像头人物 Review（通用） | 11 类 |

分类全集 11 个：`home_arrival`、`home_departure`、`package_delivery`、`food_delivery`、`cleaning`、`maintenance`、`visitor`、`elevator_activity`、`suspicious_activity`、`unknown_activity`、`unable_to_confirm`。

### 适配你自己的摄像头

**只换文案（不改代码）。** 在集成选项的「行为选项」里有三个字段：

| 字段 | 生效范围 | 说明 |
|---|---|---|
| 场景描述 | **仅 `review_six`** | 说明画面里哪个门是入户门、哪里是电梯、镜头外有什么。不写这段时，模型会把画面里最近的门当成「入户门」——实测本部署曾把「走出电梯后离开」判成 `home_departure`。 |
| 标签 | 场景契约 | 每行一条「标签: 定义」。**非空时会全局替换允许集合**，必须列出全部 11 个标签；漏掉任何一个它就永远报不出来，而且没有任何报错。 |
| 判断规则 | 场景契约 | 替换内置规则全文。契约（JSON 结构）与现场布局仍会自动附加，因为解析器依赖它们。 |

**新增一个场景（需要改代码）。** 仅注册 `Scene` **不足以**让新场景投入使用，必须同时改两处：

| 要做的事 | 位置 |
|---|---|
| 声明场景（分类、提示词、信号、版本） | `scenes.py` 的 `SCENES` |
| **让活动能路由到它**，以及该场景的抽帧计划 | `media.py` 的 `plan_evidence()` |

原因是 mode 由 `plan_evidence()` 决定，场景注册表不参与选路。因此**注册了但没有路由的场景是死代码**：它能渲染提示词，却没有任何活动会走到它。核心的抽帧、联系图、调用、交付、幂等确实无需改动，但路由是第二个必要改动点。新场景要自己负责提示词质量与验证，并比照 `review_six` 的方式自行验收。

### 实体

| 实体 | 说明 |
|---|---|
| `event.<name>_activity` | 活动完成 / 失败事件 |
| `sensor.<name>_last_classification` | 最近一次分类 |
| `sensor.<name>_last_confidence` | 最近一次置信度 |
| `sensor.<name>_last_error` | 最近一次稳定错误码 |
| `sensor.<name>_pending_activities` | 未结束的活动数 |
| `binary_sensor.<name>_healthy` | 运行时健康状态 |

**服务**：

| 服务 | 用途 |
|---|---|
| `frigate_vision.get_activity` | 返回脱敏后的活动摘要（带响应数据，含 `provider` 字段标明本条由哪组回答） |
| `frigate_vision.process_review` | 手动把某条未处理的 person Review 入队 |
| `frigate_vision.retry_failed` | 对**可证明安全**的失败建立一次显式重试 |
| `frigate_vision.ack_delivery` | 通知蓝图处理成功后回执，活动转为 `completed` |

`retry_failed` 接受：任何 5xx（`provider_http_500`…`599`）、`provider_http_429`（配额用尽）、`provider_unavailable`、`evidence_incomplete`、`frigate_unavailable`、`media_retry_exhausted` 等。**拒绝** `analysis_outcome_unknown` 与 `delivery_outcome_unknown`——模型可能已计费、通知可能已发出，重放会重复打扰。

**交付事件** `frigate_vision_activity` 字段：`entry_id`、`activity_id`、`delivery_attempt_id`、`classification`、`description`、`confidence`、`evidence_url`、`evidence_image_url`、`evidence_offsets`、`clip_url`、`hls_url`、`frigate_review_url`、`review_ids`、`occurred_at`。

其中：

- `evidence_url` 是 `media-source://` 标识，供 Home Assistant 媒体浏览器使用。
- `evidence_image_url` 是**已签名**的相对 HTTP 地址，供回放弹窗内的联系图显示。`<img>` 带不上 `Authorization` 头，HA 的鉴权中间件也没有 cookie 通道，因此未签名的 `/api/` 图片必然 401；签名与路径精确绑定，也不能复用视频的签名。
- `evidence_offsets` 是联系图**每格**对应的视频秒数（相对片段起点，已夹取到窗口内），**竖线**分隔，如 `5.3|11.2|48.1`。分隔符不能是逗号：HA 的原生模板解析器会把逗号分隔的模板结果当作 tuple，导致通知脚本以 `TypeError: TupleWrapper is not JSON serializable` 失败，整条通知都写不进去。

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

**Frigate**

- base URL、认证方式（`none` / `native`）、MQTT topic 前缀（默认 `frigate`）、精确的摄像头名。
- 端口 5000 是无认证内部 API，只应在可信内网使用；端口 8971 使用原生认证，填写用户名/密码后由共享 CookieJar 登录并自动刷新。
- zone 分组（near / transition / far）用于**过滤**哪些 Review 值得分析（配合「分析所有远端 Review」开关）。三组不得重叠。

**视觉 Provider**

- base URL、API Key、模型名。可从预设（DeepSeek / Gemini / GLM / OpenAI）选择以预填 URL，也可直接手填任意 OpenAI 兼容端点。
- 推理档位 `default` / `low` / `high` / `max`；推理开关 `default` / `disabled`。
- 保存前可点「测试连接」实测一次往返（纯文本，不发送图片）。
- **`thinking` 字段并非所有端点都接受。** Google 的 OpenAI 兼容端点会以 HTTP 400 拒绝该字段（`Unknown name "thinking"`），因此集成对已知会拒绝的 provider 自动省略此字段——此时推理保持 provider 默认，代价更高但分析能正常完成。`reasoning_effort` 不受此限制。

### 第二组 Provider（故障转移，可选）

同一页最下方有五个「备用」字段：接口地址、API Key、模型名称、推理模式、推理强度。用途是**第一组拿不到答案时不要丢掉这条活动**——实测本部署曾因第一组返回 `503 model service info not found` 而整条活动被记为失败、静默丢弃。

| 规则 | 说明 |
|---|---|
| 生效条件 | 地址 / Key / 模型**三项齐全**才启用。只填其中一两个会被表单拒绝（`fallback_incomplete`）；一项都不填＝不启用，这是默认状态。 |
| 只改下拉框不算配置 | 推理模式与推理强度有默认值，前端每次保存都会回传它们，所以「只动了下拉框」不是半配置。 |
| 共享的字段 | 提示词相关的设置（场景描述、标签、判断规则、图片宽度、输出语言等）两组**共用一份**——换供应商不该改变问题本身。每组独立的只有"谁来回话"：地址、Key、模型、推理两项。 |
| 何时转移 | 第一组**重试耗尽后**仍失败才换组；两组各自跑满 4 次（退避 2s/6s/18s），最坏约 52 秒。 |
| 什么会转移 | 服务端故障（5xx）、配额用尽（429）、连不上，以及**答了但不合契约**（分类越界、空回复、推理撑爆 token）。 |
| 什么不会转移 | 本地错误（证据缺失、状态冲突、标签格式错、图片损坏、未配置客户端）——第二组会撞同一面墙，转移纯属浪费。`analysis_outcome_unknown` 也不转移：请求可能已到达第一组并被计费。 |
| 契约类不重试 | 「答了但不合契约」直接换组，不对同一组重试 4 次：同一提示词、同一模型，重试大概率得到同样的越界答案。 |
| 每条活动独立 | 不粘住备用：下一条活动仍从第一组开始，抖动恢复后自动回主。 |
| 缓存键不含 provider | 同一条活动不会因为换组而重复计费。代价：两组模型不同时，不同活动的描述风格可能不完全一致。 |
| 告警 | **只有两组都失败**才报 Repair。救回来时不报——避免"通知正常但设置里堆着错误告警"的矛盾；日志里有 WARNING 记录两组各自的原因。两组都失败时落库的是**最后一组**的错误码，完整追溯在两组的日志里。 |

排查时想知道"这条到底是哪组答的"，用 `frigate_vision.get_activity` 的 `provider` 字段（`primary` / `fallback`，未分析为 `null`）。

**行为选项**

| 字段 | 默认 | 说明 |
|---|---|---|
| 图片宽度 | 768 | 发给模型前缩放到的宽度上限（512–1920） |
| 最大 Token | 4000 | 每次模型调用的上限（1–20000） |
| 输出语言 | `zh-CN` | 提示词与描述的语种 |
| 最短活动时长 | 10 | 短于此秒数的 Review 直接丢弃（0–120），实测 233 条 Review 中 9% 短于 10 秒 |
| 队列上限 | 10 | 每个配置项的事件队列深度 |
| 历史保留天数 | 30 | 活动记录保留期 |
| 媒体保留天数 | 7 | 联系图文件保留期 |
| 夜间未知也分析 | 开 | **当前未生效**，见下方「已知问题」 |
| 分析所有远端 Review | 开 | 关掉后只分析落在已配置 zone 内的 Review |

**人物特写（子菜单）**

| 字段 | 默认 | 说明 |
|---|---|---|
| 人物特写栏 | 关 | 在联系图下侧附加一栏人物放大图。九宫格里的人物只有约 59×74 px，辨认衣着基本不可能；特写可到 255–415 px。打开会改变发给模型的图并改变缓存键，因此默认关。 |
| 人脸检测服务 URL | 空 | 可选，用来挑**哪一帧**做特写。留空则按「检测框面积最大」选帧。它是提升项而非依赖：服务没配、连不上、超时或答案不对时一律退回该规则。 |

## 通知蓝图

导入地址：

```text
https://github.com/kaattz/ha-frigate-vision/blob/main/blueprints/automation/frigate_vision/activity_notification.yaml
```

蓝图监听 `frigate_vision_activity` 事件，负责静音时段、标题正文格式、证据图与录像链接，并在动作执行后调用 `ack_delivery`。集成本身**不硬编码任何 notify 服务**。

> **蓝图不按分类过滤。** 早期版本提供一个可勾选的分类白名单，但它是一份**分类列表的静态副本**：集成新增分类后，已保存的勾选不会自动更新。HA 的 `BlueprintInputs.validate()` 只拒绝*缺失*的 input，多出来的 input 不报错，于是事件照常分析、照常投递，却在蓝图里被**静默丢弃**——实测某天上午 5 条活动只送达 2 条，丢失的 2 条都是后来新增的分类。
>
> 现在只保留静音时段：集成本身已通过 `min_review_seconds` 与 `analyze_all_far_reviews` 决定什么值得分析，再放一份手工维护的清单只会多一条静默丢事件的路径。若你确实想按分类分流，请在自己的自动化动作里判断 `trigger.event.data.classification`。

蓝图内附带的录像链接走 HA 自身的鉴权代理（`/api/frigate_vision/clip/<entry_id>/<activity_id>.mp4`），因此外网也能打开，且不会把 Frigate 暴露到公网。

## 隐私与数据

- 联系图保存在 HA 媒体目录的 `frigate_vision/<entry_id>/<activity_id>.jpg`，通过 `media-source://frigate_vision/...` 或需登录的 `/api/frigate_vision/media/...` 端点访问，**不会**作为静态文件公开。
- 联系图会上传给**你自己配置的**视觉 Provider。请确认该 Provider 的隐私条款可接受。
- 诊断信息（`diagnostics.py`）已脱敏：不含 API Key、图片、本地路径、原始 MQTT 报文和完整模型回复。
- 仓库 `.gitignore` 明确排除 `evidence-review/` 与 `.e2e-media/`——真实画面证据**永远不要提交**。

## 故障排查

集成所有失败都带稳定错误码，可从配置项下载诊断信息定位问题。

| 错误码 | 含义与处理 |
|---|---|
| `analysis_outcome_unknown` | 模型可能已收到请求，**禁止自动重试** |
| `delivery_outcome_unknown` | 通知可能已发出，**不要补发** |
| `provider_http_5xx` | 视觉服务端故障。502/503/504 会自动重试（共 4 次尝试，退避 2s/6s/18s），因为这三者都表示请求未被处理；500 不自动重试（无法确定请求是否已执行）。耗尽后可用 `retry_failed` 手动补跑。 |
| `provider_http_429` | 配额用尽。**刻意不自动重试**：窗口通常是按天计的，几十秒的退避熬不过去，重试只会烧掉下一个窗口的额度。等配额重置后手动 `retry_failed`，或改用付费方案 / 换 provider。 |
| `provider_unavailable` | 完全连不上 provider。可用 `retry_failed` 补跑。 |
| `invalid_llm_response` | 模型答了，但分类越界或 JSON 结构不合契约。配了第二组会直接转过去（不重试）。 |
| `empty_provider_response` | 模型返回空内容。配了第二组会转过去。 |
| `reasoning_exhausted_max_tokens` | 推理把 token 预算耗尽，没留下正文。提高「最大 Token」或降低推理档位；配了第二组会转过去。 |
| `media_retry_exhausted` | 取帧/录像重试耗尽（Frigate 侧问题，**不是** provider 问题，因此不会转移）。可安全使用 `retry_failed` |
| `ambiguous_review_ownership` | Review 归属不唯一，检查 detection ID 与 zone 布局 |
| `invalid_path_data` | Frigate 路径数据非法，永久失败，不回退像素差分 |

provider 类故障会在「设置 → 修复」中报一条修复卡片，并在日志留下一条 WARNING；下一次分析成功时自动清除。具体状态码仍在 `sensor.<name>_last_error` 上。

卡片是**哪一条**取决于该错误码有没有自己的专用卡片：

- 有专用卡片的（`provider_unavailable`、`frigate_unavailable`、`media_cleanup_failed`、`storage_corrupt` 等）显示自己的卡片——名字更具体，指向也更准。
- 没有专用卡片的 provider 故障（`provider_http_503`、`provider_http_429`、`invalid_llm_response` 等）归到共享的 `provider_error` 一条，避免为每个状态码各写一份翻译。

**「要不要换一家再试」与「这是谁的问题」是两个不同的问题，由两个不同的谓词回答**，偏置刻意相反：

| 问题 | 偏置 | 理由 |
|---|---|---|
| 值得换第二组再试吗？ | **允许清单**：只要不是本地错误就试 | 未知故障宁可多试一次——不试就是丢活动 |
| 该归咎于 provider 吗？ | **闭集**：只有确实由 provider 产生的错误码才归咎 | 上报入口接收的是「代码库里任意异常的字符串」，把存储冲突或 Frigate 认证失败算成 provider 故障，会把人引到错误的子系统 |

早期版本让一个谓词同时回答两个问题（用「非本地即 provider」的补集），结果是 store / media / frigate 层抛出的 50 个错误码全部被报成「视觉 provider 正在出错」——包括 `terminal_activity`（存储冲突）和 `authentication_failed`（Frigate 认证）。代码注释与测试现在把两个谓词相反的偏置钉在一起。

**备用地址写错会怎样？** 早期版本会把它变成 `analysis_outcome_unknown`——既不报修、也不能手动补跑的一次静默丢失，比不配备用还糟。现在 URL 类错误被正确归类为 `provider_unavailable`：可转移、有卡片、可 `retry_failed` 补跑，日志里还会同时写下主组和备用组各自的原因。表单也会在保存时校验备用地址（`fallback_invalid_url`），所以手滑少写 `https://` 会在表单上就被拦下。

> **已知遗留（早于本功能）**：`provider_unavailable` 有自己的卡片，但没有任何地方调用 `clear_error("provider_unavailable")`，所以故障恢复后那张卡片可能一直留着。它不影响活动处理，但会让你在「修复」里看到过期提示。

## 已知问题

- **「夜间未知也分析」选项当前不生效。** 它在配置界面中存在并会被保存，但代码中没有任何地方读取它（`analyze_night_unknown` 仅出现在 schema 定义处）。切换它不会改变行为。
- 新增场景需要同时改 `scenes.py` 与 `media.py`（见[适配你自己的摄像头](#适配你自己的摄像头)）；仅注册 `Scene` 不会让场景可用。
- **`provider_unavailable` 的修复卡片可能残留。** 见上方「故障排查」末尾的说明：它有自己的卡片，但没有代码在故障结束后清除它。

## 升级到 0.1.0（单一流水线）

**此版本删除了门周期、door 场景与三档处理模式，且不兼容旧存档。** 部署前必须手动删除 store 文件，否则集成启动失败：

```text
/config/.storage/frigate_vision.<entry_id>
```

这会丢失全部活动历史。原因：存档加载是全有或全无的，旧记录携带的字段（`door_cycle` 来源、`processing_mode`、门铃时间等）已不存在，任何一条读不出都会让整个配置项拒绝启动。曾提供「保留死字段兼容旧记录」的方案并被否决——删除是明确的选择。

同时移除的还有：

- `select.<name>_processing_mode` 实体与「处理模式」选项。引用该实体的自动化会失效，需先改掉。
- 配置页的「门口设备」步骤。旧 entry 数据里残留的 `door` 键会被无视，不影响启动。
- Repair 告警 `door_mapping_invalid`、`door_open_too_long`。

## 兼容性

| 组件 | 支持基线 |
|---|---|
| Home Assistant | 2026.8.0+（CI 固定 2026.8.3；实测运行于 2026.9.4） |
| Frigate | 0.17.2 公开 HTTP / MQTT 契约 |
| Python | 3.14.2+ |
| 视觉 Provider | 兼容 OpenAI Chat Completions 的接口 |

集成只使用 Frigate 的公开 HTTP API 与 MQTT topic、HA 的公开扩展 API（`Store`、`mqtt.client.async_subscribe`、`media_source`、`issue_registry`），不 import 任何上游集成的内部模块。

## 开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install homeassistant==2026.8.3 pytest pytest-asyncio pytest-homeassistant-custom-component ruff mypy pillow pyyaml

python -m pytest -q                          # 30 个测试模块
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
- 不内置除 `review_six` 之外的场景实现；新增场景由使用者自行撰写并验收。
- 不做门锁 / 门磁 / 门铃事件编排——那是门周期状态机的工作，已整体移除；活动边界一律来自 Frigate Review。

## License

[MIT](LICENSE)
