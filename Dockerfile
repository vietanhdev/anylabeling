# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/anylabeling \
    ANYLABELING_DEVICE=CPU \
    QT_QPA_PLATFORM=xcb \
    QT_XCB_NO_MITSHM=1

# Runtime libraries for the PyQt6 X11 platform plugin and software rendering.
RUN dpkg --configure --pending \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/anylabeling
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY anylabeling ./anylabeling
# Install this checkout with its normal CPU dependencies. The BuildKit cache
# speeds up rebuilds without retaining pip's download cache in the image.
# PIP_CACHE_DIR also applies to pip's isolated build subprocesses.
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_CACHE_DIR=/root/.cache/pip python -m pip install .

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN test "$USER_ID" -ne 0 \
    && test "$GROUP_ID" -ne 0 \
    && groupadd --gid "$GROUP_ID" anylabeling \
    && useradd --uid "$USER_ID" --gid "$GROUP_ID" \
        --create-home --no-log-init anylabeling \
    && mkdir /data \
    && chown anylabeling:anylabeling /data

USER anylabeling
WORKDIR /data
ENTRYPOINT ["anylabeling"]
