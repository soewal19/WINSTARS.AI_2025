FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY . /app
EXPOSE 8888
CMD ["bash","-lc","pytest -q || true; jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
