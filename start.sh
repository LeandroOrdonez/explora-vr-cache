#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    CIP=$(getent hosts ${CLIENT_HOST} | cut -d' ' -f1)
    echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}"
    ./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}
fi

service redis-server start
echo "Loading Prefetcher..."
nohup python -u ./instance/prefetcher.py >/dev/null 2>&1 &
echo "Starting NGINX..."
service nginx start
uwsgi --ini uwsgi.ini