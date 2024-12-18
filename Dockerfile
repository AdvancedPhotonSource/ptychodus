#FROM python:3.12-slim-bullseye
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y libqt5gui5 && \
    rm -rf /var/lib/apt/lists/*
#ENV QT_DEBUG_PLUGINS=1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the wheel file into the container at /app
COPY dist/ptychodus-*.whl dist/ptychi-*.whl .

# Install the wheel
RUN python3 -m pip install --no-cache-dir --find-links=. ptychodus[globus,gui] ptychi && \
    rm ptychodus-*.whl ptychi-*.whl

# Run ptychodus when the container launches
CMD ["python3", "-m", "ptychodus"]
