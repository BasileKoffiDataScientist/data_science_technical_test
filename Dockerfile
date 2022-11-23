From continuumio/miniconda3

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME


RUN conda update --name base conda &&\
    conda env create --file servier.yaml
SHELL ["conda", "run", "--name", "api", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--name", "api", "python", "src/main.py"]

