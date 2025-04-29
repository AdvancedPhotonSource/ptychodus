#FROM python:3.12-slim-bullseye
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y git libqt5gui5 && \
    rm -rf /var/lib/apt/lists/*
#ENV QT_DEBUG_PLUGINS=1

# Upgrade pip
RUN python3 -m pip install --root-user-action=ignore --no-cache-dir --upgrade pip

# Build the wheel
WORKDIR /src

# Copy the sources
COPY . .

# Set the working directory in the container
WORKDIR /app

# Install the wheel
RUN python3 -m pip install --root-user-action=ignore --no-cache-dir /src[globus,gui,ptychi] \
    && rm -rf /src

# Run ptychodus when the container launches
CMD ["python3", "-m", "ptychodus"]
