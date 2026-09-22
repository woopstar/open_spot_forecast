@echo off
ssh root@192.168.123.9 -p 22000 "rm -rf /config/custom_components/open_spot_forecast/ && mkdir -p /config/custom_components/open_spot_forecast/"
tar --exclude="__pycache__" --exclude="*.pyc" -C custom_components/open_spot_forecast -cf - . | ssh root@192.168.123.9 -p 22000 "cd /config/custom_components/open_spot_forecast && tar xf -"
