FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set bash as default shell
ENV SHELL=/bin/bash

# Build with some basic utilities


RUN apt-get update && apt-get install -y --no-install-recommends python3
RUN apt-get install -y sudo
#RUN apt-get install -y libgl1-mesa-dev

#RUN pip uninstall llvmlite
#RUN conda uninstall llvmlite
#RUN apt-get uninstall llvmlite

#RUN apt-get install -y nodejs npm

RUN pip install --no-cache-dir --upgrade pip wheel setuptools
#RUN conda update -n base -c defaults conda


#RUN conda install -c conda-forge nodejs
#RUN pip install llvmlite --ignore-installed
RUN pip install --no-cache-dir \
    opencv-python \ 
    #opencv-contrib-python \
    #opencv-python-headless \    
    scikit-image \
    jupyterlab \
    ipywidgets \
    # monai \
    # monailab-weekly \
    shap \
    # grad-cam \ 
    # pytorch-lightning \
    # timm \
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
