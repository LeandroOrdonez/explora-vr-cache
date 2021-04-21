#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    CIP=$(ip r | grep default | awk -v OFS=\| '{ print $3 }')
    echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}"
    ./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}
fi

service redis-server start
echo "Loading Prefetcher..."
nohup python -u ./instance/prefetcher.py > prefetcher.log &
echo "Starting NGINX..."
service nginx start
uwsgi --ini uwsgi.ini