FROM python:3.12-alpine
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apk update && apk add --no-cache \
    g++ gcc git hdf5-dev linux-headers musl-dev pkgconfig python3-dev py3-qt5 py3-yaml

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container at /app
COPY . .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN python3 -m pip install --no-cache-dir .

# Set environment variable to use host's X11 display
ENV DISPLAY=:0

# Run ptychodus when the container launches
ENTRYPOINT ["python", "-m", "ptychodus"]
