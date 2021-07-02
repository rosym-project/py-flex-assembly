#!/usr/bin/env python3

import pyrealsense2 as rs

pipeline = rs.pipeline()

try:
    pipeline.start()
    pipeline.stop()
except RuntimeError:
    exit(1)
    
exit(0)
