Install with `docker build -t --ulimit nofile=1024 -t cse598 .`;
Run with `docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cse598`

If you're running Windows, its better to just install openrave with their compiled binaries from https://sourceforge.net/projects/openrave/files/latest_stable/
