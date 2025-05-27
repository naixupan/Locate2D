'''
文件名称：Locator_dev.py
作者：何广鹏
日期：2025年5月26日
功能：2D相机定位，手眼标定集成
版本：V0.0
'''

import json
import cv2
import numpy as np
from utils_dev import *
import logging
import datetime

error_code = 200
error_message = ""

logging.basicConfig(
    filename="locator_dev.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class Locator:
    def __init__(self):
        pass
    def calibrate_camera(self):
        pass
    def calibrate_eye_in_hand(self):
        pass
    def locate(self):
        pass