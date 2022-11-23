FROM continuumio/miniconda3

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version

RUN conda update -n base conda

# Create the environment:
#COPY servier.yml .
#RUN conda env create -f servier.yml

RUN conda create -y --name servier python=3.6
# RUN conda create -n servier -c conda-forge -c bioconda python==3.6 pip

#RUN conda activate servier
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "servier", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"

WORKDIR /api

COPY api .

COPY test .

RUN conda install -c conda-forge rdkit

COPY requirements.txt .

RUN conda install --file requirements.txt










