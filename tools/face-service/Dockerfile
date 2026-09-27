# A tiny face detector, so the close-up shows the person's face when one is visible.
#
# Deliberately a separate service rather than a dependency of the integration. The
# HA container is Alpine (musl libc) and every OpenCV wheel on PyPI is built for
# glibc, so opencv cannot be pip-installed there at all -- verified: pip reports no
# matching distribution. Adding it to the integration would mean compiling OpenCV
# against musl inside the HA image, which breaks on every HA upgrade.
#
# Here it is a plain glibc image, where the wheel installs in seconds.
FROM python:3.12-slim

# opencv-python-headless: no GUI libraries, which is all this needs. Pinned to 4.x
# rather than 5.x: 5.0 removed cv2.CascadeClassifier, and 4.12 is a known-good
# release with the same abi3 wheel.
RUN pip install --no-cache-dir "opencv-python-headless==4.12.0.88"

WORKDIR /app

# The YuNet detector, downloaded at build time so the container needs no network at
# run time. 232 KB. Not vendored into the repository: it is a third-party binary,
# and a build-time fetch keeps the provenance explicit.
ADD https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx /app/yunet.onnx

COPY server.py /app/server.py

EXPOSE 8788

# One worker: detection is CPU-bound and short, and a single process keeps the
# per-size detector cache simple.
CMD ["python", "-u", "server.py"]
