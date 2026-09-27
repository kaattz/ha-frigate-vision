# 人脸检测服务（Docker）

给「人物特写」选帧用：从同一段录像的若干候选帧里，挑出能看到正脸的那一帧。

## 为什么是独立容器

**不是**因为性能，而是因为 HA 容器装不了 OpenCV：

| 环境 | libc | `pip install opencv-python-headless` |
|---|---|---|
| HA 容器（Alpine 3.24） | **musl** | ❌ 报 `No matching distribution` |
| 普通 Linux 容器（Debian slim） | **glibc** | ✅ 直接装 wheel，约 40 秒 |

PyPI 上所有 `opencv-python-headless` 的 Linux wheel 都是 `manylinux`（glibc）。
Alpine 的 `apk` 有原生 `py3-opencv`，但它会拉进第二个 Python（105 个包），
和 HA 自己编译的 `/usr/local/bin/python3.14` 冲突。

独立容器同时满足了另外两点：不碰 Frigate 主机、不碰 HA 镜像，升级互不影响。

## Coral TPU 能不能用上

**不能，而且这个任务也不需要。**

OpenCV 官方源码里 `enum Backend` 的完整枚举是：

```
DNN_BACKEND_DEFAULT / HALIDE / INFERENCE_ENGINE / OPENCV /
VKCOM / CUDA / WEBNN / TIMVX / CANN
```

**没有 EdgeTPU**。Coral 的官方路径是 `pycoral` + `tflite-runtime`（或 TensorFlow Lite
的 edgetpu 委托），和 OpenCV 是两条独立的栈。`pycoral` 在 PyPI 上也没有
Linux x86_64 wheel。

不需要，因为实测：

```
单次人脸检测（含 HTTP 往返）: 24 ms
一次活动 6 个候选帧        : 144 ms
```

这是一次**活动结束后**的批处理，不是实时路径 —— 144 毫秒相对于一次
几秒的 VLM 分析可以忽略。Coral 的价值在于 Frigate 那种「每路摄像头每秒多帧」
的场景（当前 6.5 ms/帧），用在 6 张图上没有意义。

## 接口

```
GET  /health
     -> {"status":"ok","model":"yunet.onnx","threshold":0.6}

POST /face
     body: {"image": "<base64 JPEG>"}
     回:   {"has_face": true, "score": 0.892, "box": [x,y,w,h]}
       或  {"has_face": false, "score": 0.0, "box": null}
```

**送人物裁剪图，不是整帧。** 整帧 637×360 里人头只有约 30 px；裁剪图能让同样的
带宽覆盖更多脸部像素。实测差异很大：整帧 0/6，裁剪图 6/6。

## 运行

```bash
docker build -t frigate-vision-face .
docker run -d --name frigate-vision-face \
  --restart unless-stopped -p 8788:8788 frigate-vision-face
```

## 实测结果（真实活动，12 个候选帧）

真值由我逐帧看图确定（正面 6 帧 / 背面 6 帧）：

| | 命中 |
|---|---|
| 正面帧检出 | **6/6** |
| 背面帧误报 | **0/6** |

阈值 0.6 是量出来的分界：全部正面帧得 0.688–0.893，唯一的背面误报得 0.523。

## 集成侧的用法（提升项，不是必须项）

```
1. 先按现有逻辑选出【面积最大】的 detection           <- 永远可用
2. 若人脸服务可达: 对候选帧各问一次，改用【检出人脸】的
3. 服务不可达 / 超时 / 任何异常 -> 静默退回第 1 步
```

第 3 条是硬要求：这个服务**不允许成为关键路径**。没有它时行为必须和现在完全一样。
