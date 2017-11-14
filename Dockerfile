# To facilitate file sharing between a user on the host and container, the 
# entrypoint script of the container will scan /home/user and set the uid and 
# gid of user accordingly. As a result you can mount any volume from the host 
# in /home/user and edit the files from the host with permission conflicts.
#
# Example:
#   nvidia-docker build --rm -t my_torch_container .
#   nvidia-docker run --interactive --publish 8888:8888 \
#       --volume /home/johnsmith/human-pose-estimation:/home/user/human-pose-estimation \
#       my_torch_container


FROM pixelou/nvidia-torch:extra-latest


ADD . /home/user/human-pose-estimation/
RUN chown user:user /home/user/human-pose-estimation/ -R
