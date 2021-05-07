#!/usr/bin/env bash

if [ "$ENABLE_TC" == "true" ]
then
    for h in 1 2 3 4 5 6
    do
        CIP=$(getent hosts ${CLIENT_HOST}_${h} | cut -d' ' -f1)
        echo "./traffic-control.sh -o --delay=${LATENCY} --jitter=${JITTER} --uspeed=${BANDWIDTH} --dspeed=${BANDWIDTH} ${CIP}"
        tc qdisc add dev eth0 root handle 1a1a: htb default 1
        tc class add dev eth0 parent 1a1a: classid 1a1a:${h} htb rate ${BANDWIDTH}
        tc qdisc add dev eth0 parent 1a1a:${h} handle 220${h}: netem delay ${LATENCY}ms
        tc filter add dev eth0 protocol ip parent 1a1a: prio 1 u32 match ip dst ${CIP} flowid 1a1a:${h}
    done
fi

service redis-server start
echo "Loading Prefetcher..."
nohup python -u ./instance/prefetcher.py > prefetcher.log &
echo "Starting NGINX..."
service nginx start
uwsgi --ini uwsgi.ini