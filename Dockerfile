FROM python:3.7-slim

ENV FLASK_APP="run.py"
ENV APP_SETTINGS="production"
ENV QUERY_LOG="./issued_queries.log"
ENV ENABLE_PREFETCHING="true"
ENV VIEWPORT_SIZE=-1
ENV BUFFER_SIZE=4
ENV SERVER_URL="http://localhost:80"
ENV SERVER_HOST="localhost"
ENV CLIENT_HOST="localhost"
ENV PERFECT_PREDICTION="true"
ENV REDIS_HOST="localhost"
ENV ENABLE_TC="false"
ENV BANDWIDTH="25Mbps"
ENV LATENCY=15
ENV JITTER=2

ARG gid=www-data
ARG uid=www-data

WORKDIR /app

ADD . /app

RUN chown -R ${uid}:${gid} /app

RUN apt-get clean \
    && apt-get update 
    
RUN apt-get install -y nginx gcc g++ openssh-server htop build-essential python-dev python3-dev iproute2 inetutils-ping redis-server kmod iperf3 bmon
RUN pip install -r requirements.txt --src /usr/local/src

# RUN tc qdisc add dev lo root handle 1: htb default 12
# RUN tc class add dev lo parent 1:1 classid 1:12 htb rate ${BANDWIDTH}
# RUN tc qdisc add dev lo parent 1:12 netem delay ${LATENCY} ${JITTER} distribution normal

COPY nginx.conf /etc/nginx
RUN chmod +x ./traffic-control.sh
RUN chmod +x ./start.sh
# RUN chmod +x ./instance/prefetcher.py
CMD ["./start.sh"]
#CMD ["flask", "run", "--host", "0.0.0.0"]