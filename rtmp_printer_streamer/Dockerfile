ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# Packages: ffmpeg for streaming, curl for health checks
RUN apk add --no-cache \
    ffmpeg \
    curl \
    python3 \
    python3-dev \
    py3-pip \
    py3-numpy \
    py3-opencv \
    gstreamer \
    gst-plugins-base \
    gst-plugins-good \
    gst-plugins-bad \
    gst-plugins-ugly \
    gstreamer-tools

# S6 overlay service and init scripts
COPY rootfs /

# Ensure scripts are executable
RUN chmod +x /etc/services.d/streamer/run \
    && chmod +x /etc/cont-init.d/10-config