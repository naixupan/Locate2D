import json

import cv2
import numpy as np
from cv2.typing import Range

from utils_dev import *
import logging
import datetime
import time

import pandas as pd
import os

error_code = 200
error_message = ""

logging.basicConfig(
    filename="locate_aruco.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def locate_aruco_marks(user_imput):

    global error_code, error_message
    result_data = {}
    time_begin = datetime.datetime.now()

    try:
        config = json.loads(user_imput)
        config = eval(config)

    except Exception as e:
        logging.error(e)
        error_code = 401
        error_message = 'Your input is not a dictionary ! '
        return result_data

    logging.info(f'输入参数：{config}')
    try:
        locate_type = config['locate_type']
        image_path = config.get("image_path")
        robot_poses = config.get("robot_pose")
        camera_prameter_path = config.get("camera_prameter_path")
        hand_eye_prameter_path = config.get("hand_eye_prameter_path")
        save_folder = config.get("save_path")
        relative_position_path = config.get("relative_position_path")
        original_coords = config.get("original_coords")
        adjust_center = config.get("adjust_center")

        if locate_type == '1':
            relative_position_path_1 = config.get("relative_position_path_1")

    except Exception as e:
        logging.error(f'[{e}')
        error_code = 402
        error_message = 'Your input is invalid ! '
        return result_data
    
    K , distCoeffs = load_camera_parameters(camera_prameter_path)
    
    cam2gripperMatrix = load_calibrate_matrix(hand_eye_prameter_path)

    target2markMatrix = load_calibrate_matrix(relative_position_path)

    if locate_type == '1':
        target2markMatrix_ = load_calibrate_matrix(relative_position_path_1)

    image_path_list = image_path.split('/')
    aruco_image_path1 = f'{image_path}-1.bmp'
    aruco_image_path2 = f'{image_path}-2.bmp'
    aruco_image_path3 = f'{image_path}-3.bmp'
    save_name = f'{image_path_list[-1]}.png'
    save_path = f'{save_folder}/{save_name}'

    logging.info(f'image_path1:{aruco_image_path1}')
    logging.info(f'image_path2:{aruco_image_path2}')
    logging.info(f'image_path3:{aruco_image_path3}')

    aruco_image_path = [aruco_image_path1, aruco_image_path2, aruco_image_path3]
    pose_list = robot_poses.split(' ')
    # print(pose_list)
    robot_pose_num = []
    original_coords_num = []

    # for pose in pose_list:
    #     pose_num = float(pose)
    #     robot_pose_num.append(pose_num)

    for i in Range(6):
        robot_pose_num_ = float(pose_list[i])
        original_coords_num_ = float(original_coords[i])
        robot_pose_num.append(robot_pose_num_)
        original_coords_num.append(original_coords_num_)


    gripper2base = Pose2HomogeneousMatrix(robot_pose_num)
    
    avg_pose = [0, 0, 0, 0, 0, 0]
    target2camera, corners = compute_aruco_pose(aruco_image_path[0], K, distCoeffs, 0.03, True, save_path)
    ############ 中心点位置判断
    at_center_ = False
    # 第一次调整中心点位置
    if locate_type == '1' and adjust_center == '0':
        logging.info(f'未调整过Mark点中心点位置')
        camera2base = gripper2base @ cam2gripperMatrix
        target2base = gripper2base @ cam2gripperMatrix @ target2camera
        camera2base[0][3] = target2base[0][3]
        camera2base[1][3] = target2base[1][3]
        gripper2base_ = camera2base @ np.linalg.inv(cam2gripperMatrix)
        target_pose_ = HomogeneousMatrix2Pose(gripper2base_)

    # 调整位置后检查
    elif locate_type == '1' and adjust_center == '1' :
        image_ = cv2.imread(aruco_image_path[0])
        image_gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        at_center_ = at_center(image_gray, corners)
        if at_center_ == False:
            # logging.info(f"相机在Mark点坐标系下位姿：\n{target2camera}")
            logging.info(f'Mark点中心点未在中心范围内，需要移动')
            # 修改相机位姿的x，y的值
            camera2base = gripper2base @ cam2gripperMatrix
            logging.info(f"相机当前位姿：\n{camera2base}")
            target2base = gripper2base @ cam2gripperMatrix @ target2camera
            camera2base[0][3] = target2base[0][3]
            camera2base[1][3] = target2base[1][3]
            logging.info(f"移动后相机位姿：\n{camera2base}")
            gripper2base_ = camera2base @ np.linalg.inv(cam2gripperMatrix)
            logging.info(f"\n移动前机械臂位姿：\n{gripper2base}")
            logging.info(f"\n移动后机械臂位姿：\n{gripper2base_}")
            target_pose_ = HomogeneousMatrix2Pose(gripper2base_)

    ###########################

    target2base = gripper2base @ cam2gripperMatrix @ target2camera
    target_pose = HomogeneousMatrix2Pose(target2base)
    for i in range(6):
        avg_pose[i] = target_pose[i]

##################### 测试数据计算 #####################
    csv_path = './data_record.csv'
    inital_result = initalize_csv(csv_path,COLUMNS)
    if inital_result == 0:
        logging.info(f'已创建CSV文件：{csv_path}')
    else:
        logging.info(f'CSV文件已存在：{csv_path}')
    # 第一次定位：水平拍照位，未调整中心点
    if locate_type == '1' and adjust_center == '0' :
        # 1阶段，水平拍照位数据计算
        T_mark2base_test = Pose2HomogeneousMatrix(avg_pose)
        T_target2mark_test = target2markMatrix_
        T_target2base_test = T_mark2base_test @ T_target2mark_test
        target_pose_test = HomogeneousMatrix2Pose(T_target2base_test)
        logging.info(f'### 1阶段计算结果：\n{target_pose_test}')
        logging.info(f'### 原始数据：\n{original_coords}')
        data_type = 1

    # 第二次定位，水平拍照位，调整中心点
    elif locate_type == '1' and adjust_center == '1' :
        T_mark2base_test = Pose2HomogeneousMatrix(avg_pose)
        T_target2mark_test = target2markMatrix_
        T_target2base_test = T_mark2base_test @ T_target2mark_test
        target_pose_test = HomogeneousMatrix2Pose(T_target2base_test)
        logging.info(f'### 2阶段计算结果：\n{target_pose_test}')
        logging.info(f'### 原始数据：\n{original_coords}')
        data_type = 2

    # 第三次定位，垂直拍照位
    elif locate_type == '2':
        T_mark2base = Pose2HomogeneousMatrix(avg_pose)
        T_target2mark = target2markMatrix
        T_target2base_ = T_mark2base @ T_target2mark
        target_pose_test = HomogeneousMatrix2Pose(T_target2base_)
        logging.info(f'### 3阶段计算结果：\n{target_pose_test}')
        logging.info(f'### 原始数据：\n{original_coords}')
        data_type = 3

    single_data = {
        'x_compute': target_pose_test[0],
        'y_compute': target_pose_test[1],
        'z_compute': target_pose_test[2],
        'rz_compute': target_pose_test[5],
        'x_original': original_coords_num[0],
        'y_original': original_coords_num[1],
        'z_original': original_coords_num[2],
        'rz_original': original_coords_num[5],
        'type': data_type
    }

    if append_to_csv(csv_path, single_data):
        logging.info(f'已追加 1条数据到 {csv_path}')
######################################

    # 1表示第一次定位
    if locate_type == '1':

        if adjust_center == '0' or at_center_ == False:
            result_data["code"] = error_code
            result_data["msg"] = error_message
            result_data["at_center"] = '0'
            result_data["x"] = target_pose_[0]
            result_data["y"] = target_pose_[1]
            result_data["z"] = robot_pose_num[2]
            result_data["rx"] = robot_pose_num[3]
            result_data["ry"] = robot_pose_num[4]
            result_data["rz"] = robot_pose_num[5]

        else:
            dx = target2markMatrix[0]
            dy = target2markMatrix[1]
            dz = target2markMatrix[2]
            result_data["code"] = error_code
            result_data["msg"] = error_message
            result_data["at_center"] = '1'

            result_data["x"] = avg_pose[0] + dx
            result_data["y"] = avg_pose[1] + dy
            result_data["z"] = avg_pose[2] + dz
            result_data["rx"] = avg_pose[3]
            result_data["ry"] = avg_pose[4]
            result_data["rz"] = avg_pose[5]
    # 2表示第二次定位
    elif locate_type == '2':
        T_mark2base = Pose2HomogeneousMatrix(avg_pose)
        T_target2mark = target2markMatrix
        T_target2base_ = T_mark2base @ T_target2mark
        target_pose_ = HomogeneousMatrix2Pose(T_target2base_)
        result_data["code"] = error_code
        result_data["msg"] = error_message
        result_data["x"] = target_pose_[0]
        result_data["y"] = target_pose_[1]
        result_data["z"] = target_pose_[2]
        result_data["rx"] = target_pose_[3]
        result_data["ry"] = target_pose_[4]
        result_data["rz"] = target_pose_[5]
    
    end_time = datetime.datetime.now()
    logging.info(f"第{locate_type}阶段算法用时：{end_time - time_begin}")

    return result_data

def main():
    global  error_message
    global error_code
    while(True):
      user_input = input("> ")
      if user_input.lower() == 'exit':
        logging.info(f"exit time :{datetime.datetime.now()}")
        return
      try:
        # print('开始定位')
        result_data = locate_aruco_marks(user_input)
        if error_code != 200:
          result_data["code"] = error_code
          result_data["msg"] = error_message
          result_data["x"] = 0
          result_data["y"] = 0
          result_data["z"] = 0
          result_data["rx"] = 0
          result_data["ry"] = 0
          result_data["rz"] = 0

        print(json.dumps(result_data, ensure_ascii=False, indent=4))
        logging.info(f'输出参数：{json.dumps(result_data, ensure_ascii=False, indent=4)}')
        error_code = 200
        error_message = ''
        print("end")
      
      except Exception as e:
        logging.error(f"Error: {e}")
        result_data = {}
        result_data["code"] = error_code
        result_data["msg"] = error_message
        result_data["x"] = 0
        result_data["y"] = 0
        result_data["z"] = 0
        result_data["rx"] = 0
        result_data["ry"] = 0
        result_data["rz"] = 0
  
        print(json.dumps(result_data, ensure_ascii=False, indent=4))
        logging.info(f'输出参数：{json.dumps(result_data, ensure_ascii=False, indent=4)}')
        print("end")
        error_code = 200
        error_message = ''
    
    
if __name__ == "__main__":
  main()