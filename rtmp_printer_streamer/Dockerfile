ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG C.UTF-8

# Packages: ffmpeg for streaming, curl for health checks
RUN apk add --no-cache ffmpeg curl

# S6 overlay service and init scripts
COPY rootfs /

# Ensure scripts are executable
RUN chmod +x /etc/services.d/streamer/run \
    && chmod +x /etc/cont-init.d/10-config