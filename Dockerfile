FROM python:3.8.5

WORKDIR /app

COPY src ./src 
COPY requirements.txt setup.py README.md oracle_client.zip ./

ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y wget libaio1 unzip && \
    mkdir -p /opt/oracle && \
    cd /opt/oracle && \
    cp /app/oracle_client.zip . && \
    unzip oracle_client.zip && \
    rm oracle_client.zip

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 80

CMD uvicorn src.app.main:app --host 0.0.0.0 --port 80
