From continuumio/miniconda3


#---------------- Prepare the envirennment
RUN conda update --name base conda &&\
    conda env create --file servier.yaml
SHELL ["conda", "run", "--name", "app", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--name", "app", "python", "src/main.py"]

