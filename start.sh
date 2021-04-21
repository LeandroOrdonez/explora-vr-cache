#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    #CIP=$(getent hosts ${CLIENT_HOST} | cut -d' ' -f1)
    echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${EXPLORA_VR_CLIENT_SERVICE_HOST}"
    ./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${EXPLORA_VR_CLIENT_SERVICE_HOST}
fi

service redis-server start
echo "Loading Prefetcher..."
nohup python -u ./instance/prefetcher.py > prefetcher.log &
echo "Starting NGINX..."
service nginx start
uwsgi --ini uwsgi.ini