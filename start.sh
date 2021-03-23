#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    SIP=$(getent hosts ${SERVER_HOST} | cut -d' ' -f1)
    echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${SIP}"
    ./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} ${SIP}
fi

service nginx start
uwsgi --ini uwsgi.ini