FROM ubuntu:22.04
WORKDIR ${HOME}/
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
COPY src ./src
COPY config ./config
CMD ["python3", "app.py"]