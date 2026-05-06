# =============================================================================
# Auto Subtitle Generator — Reproducible GPU Docker Image
# =============================================================================
#
# Build:
#   docker image build --tag auto-sub-gen .
#
# Run (GPU):
#   docker run --gpus all --mount='type=volume,source=auto-sub-gen,target=/app/models' --mount='type=bind,source=/path/to/videos,target=/app/input' auto-sub-gen /app/input/video.mkv
#
# =============================================================================

FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:bcf8f5037535884fffbde1c1584af29e9eccc3f432d1cb05a5216a1184af12d8

WORKDIR /app

# cache OS dependencies
# To find the <x> version: docker container run --rm nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 bash -c "apt-get update -qq && apt-cache policy <x>"
RUN <<EOF
	set -e
	apt-get update
	apt-get install -y --no-install-recommends \
		gcc=4:13.2.0-7ubuntu1 \
		python3=3.12.3-0ubuntu2.1 \
		python3-dev=3.12.3-0ubuntu2.1 \
		python3-pip=24.0+dfsg-1ubuntu1.3 \
		ffmpeg=7:6.1.1-3ubuntu5
	rm -rf /var/lib/apt/lists/*
EOF

# stop python from erroring when installing packages system wide
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# cache pip dependencies
COPY requirements.txt .
RUN <<EOF
	set -e
	# faster-whisper depends on plain "onnxruntime", while audio-separator[gpu] depends on "onnxruntime-gpu".
	# Both share the same Python namespace — whichever installs last overwrites the other's native library.
	# If plain "onnxruntime" wins, CUDAExecutionProvider is silently unavailable.
	# To prevent this we comment out faster-whisper from requirements.txt, add its unique transitive dependencies (ctranslate2, av) in its place, then install faster-whisper separately with --no-deps.
	sed -i 's/^faster-whisper/# &/' requirements.txt
	echo 'ctranslate2==4.7.1' >> requirements.txt
	echo 'av==17.0.1' >> requirements.txt
	pip install --no-cache-dir -r requirements.txt
	pip install --no-cache-dir --no-deps faster-whisper==1.1.1
EOF
RUN pip install --no-cache-dir pytest pytest-cov

# application
COPY auto_subtitle.py .
COPY config.yaml .
COPY modules/ modules/
# transformers v4.x uses torch_dtype=, not dtype=
RUN sed -i 's/dtype=dtype,/torch_dtype=dtype,/g' modules/models.py

# test
COPY pytest.ini .
COPY tests/ tests/
RUN python3 -m pytest

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/models/huggingface

ENTRYPOINT ["python3", "auto_subtitle.py"]
