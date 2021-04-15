#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    CIP=$(getent hosts ${CLIENT_HOST} | cut -d' ' -f1)
    echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}"
    ./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}
fi

service nginx start
uwsgi --ini uwsgi.ini