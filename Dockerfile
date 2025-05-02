#FROM python:3.12-slim-bullseye
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the sources
COPY . /src

# Install & upgrade software
RUN apt-get update && \
    apt-get install -y git libqt5gui5 && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --root-user-action=ignore --no-cache-dir --upgrade pip && \
    python3 -m pip install --root-user-action=ignore --no-cache-dir /src[globus,gui,ptychi] && \
    rm -rf /src

# Run ptychodus when the container launches
#ENV QT_DEBUG_PLUGINS=1
CMD ["python3", "-m", "ptychodus"]
