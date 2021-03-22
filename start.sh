#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    ./traffic-control.sh -i -d ${LATENCY} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${SERVER_IP}
fi

service nginx start
uwsgi --ini uwsgi.ini