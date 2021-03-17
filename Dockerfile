FROM python:3.7-slim

ENV FLASK_APP="run.py"
ENV APP_SETTINGS="production"
ENV QUERY_LOG="./issued_queries.log"
ENV ENABLE_PREFETCHING="true"
ENV BUFFER_SEQ_LENGTH 5
ENV BUFFER_SIZE 20
ENV SERVER_URL "http://localhost:80"

ARG gid=www-data
ARG uid=www-data

WORKDIR /app

ADD . /app

RUN chown -R ${uid}:${gid} /app

RUN apt-get clean \
    && apt-get update 
    
RUN apt-get install -y nginx gcc g++ proj-bin libproj-dev libgeos-dev openssh-server libspatialindex-dev libpq-dev htop build-essential python-dev python3-dev 
RUN pip install -r requirements.txt --src /usr/local/src

COPY nginx.conf /etc/nginx
RUN chmod +x ./start.sh
CMD ["./start.sh"]
#CMD ["flask", "run", "--host", "0.0.0.0"]