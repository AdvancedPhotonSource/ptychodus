time docker build -t python-ptychodus .
xhost +local:docker
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix python-ptychodus
docker ps -a
sudo docker system prune -a
xhost -local:docker
