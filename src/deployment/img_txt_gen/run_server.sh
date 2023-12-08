#!/usr/bin/bash
gunicorn -t 0 \
		 -w 4 \
		 -b 0.0.0.0:12002 \
		 flask_server:app