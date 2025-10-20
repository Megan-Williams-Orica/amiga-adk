#!/bin/bash
source ~/farm-ng-amiga/venv/bin/activate
python main.py \
 --config ./configs/config.json \
 --tool-config-path ./configs/tool_config.json \
 --waypoints-path ./surveyed-waypoints/physicsLabBack2Lanes.csv \
 --last-row-waypoint-index 3 \
 --turn-direction left \
 --row-spacing 3.0 \
 --headland-buffer 2.0 \
 --actuator-enabled --actuator-id 0 --actuator-open-seconds 6 --actuator-close-seconds 7
