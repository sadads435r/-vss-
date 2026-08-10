FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY camera-entrypoint.sh /usr/local/bin/camera-entrypoint
RUN chmod 0755 /usr/local/bin/camera-entrypoint

ENTRYPOINT ["/usr/local/bin/camera-entrypoint"]
