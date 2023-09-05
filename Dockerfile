FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set bash as default shell
ENV SHELL=/bin/bash

# Build with some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends python3
RUN apt-get install -y sudo
RUN apt-get install -y libgl1-mesa-dev

#timezone
#RUN apt-get install -y --no-install-recommends tzdata
#ENV TZ=Asia/Tokyo
#RUN apt-get install -y nodejs npm

RUN conda install -c conda-forge nodejs

RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir \
    opencv-python==4.5.4.60  \ 
    opencv-contrib-python==4.5.4.60  \
    opencv-python-headless==4.5.4.60 \    
    scikit-image \
    jupyterlab \
    ipywidgets \
    monai \
    #monailab-weekly \
    #shap \
    #grad-cam \ 
    # pytorch-lightning \
    timm \
    tqdm \
    tensorboardX \
    torchsummaryX \
    torch-summary \
    six \
    seaborn
  


RUN apt-get update


ARG USERNAME=hs
ARG GROUPNAME=hs
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME -G sudo 

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo 'hs ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/


# Set bash as default shell
ENV SHELL=/bin/bash

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
