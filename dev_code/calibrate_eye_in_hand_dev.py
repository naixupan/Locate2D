'''
手眼标定，开发版本代码，参考原版代码CalibrateEyeInHand0514.py
'''

import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import math
from utils_dev import *
import logging
import datetime
import json

error_code = 200
error_message = ""

logging.basicConfig(
    filename="calibrate.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def calibrate_camera(image_folder, image_number, serial_number,pattern_size = (9,6), square_size = 0.003, image_save_path = None):
    if image_save_path is not None:
        save_path = f'{image_save_path}/calibrate_camera/{serial_number}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imagepoints = []

    for i in range(image_number):
        image_path = f'{image_folder}/image{i+1}.bmp'
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            corners_refind = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imagepoints.append(corners_refind)
            if image_save_path is not None:
                cv2.drawChessboardCorners(image, pattern_size, corners_refind, ret)
                cv2.imwrite(f"{save_path}/image{i+1}.png", image)
    ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)

    logging.info(f'相机内参：{K}')
    logging.info(f'畸变系数：{distCoeffs}')

    # 重投影误差计算
    mean_error = 0
    for i in range(len(objpoints)):
        image_points_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, distCoeffs)
        error = cv2.norm(imagepoints[i], image_points_projected, cv2.NORM_L2)/len(imagepoints[i])
        mean_error += error

    # print("Reprojection Error: ", mean_error/len(objpoints))
    logging.info(f'重投影误差：{mean_error/len(objpoints)}')
    return K, distCoeffs

def compute_mark2camera(image_folder, image_number, K, distCoeffs,serial_number,board_type='chessboard',
                        pattern_size = (9, 6), square_size = 0.02, save_corner = False, save_path=None):
    t_mark2camera = []
    R_mark2camera = []

    save_path_ = f'{save_path}/calibrate_EyeInHand/{serial_number}'

    if not os.path.exists(save_path_):
        os.makedirs(save_path_)

    for i in range(image_number):
        image_path = f'{image_folder}/image{i + 1}.bmp'
        save_path_i = f'{save_path_}/image{i + 1}.png'
        if board_type == 'chessboard':
            objp, corners, ret = DetectChessBoard(image_path, pattern_size, square_size, save_corner, save_path_i)
        elif board_type == 'circleboard':
            objp, corners, ret = DetectCircleBoard(image_path, pattern_size, square_size, save_corner, save_path_i)
        else:
            print('board_type error!')
            return None, None
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, distCoeffs)
        R_mark2cam, _ = cv2.Rodrigues(rvec)  # 旋转向量转旋转矩阵
        t_mark2camera.append(tvec)
        R_mark2camera.append(R_mark2cam)

    R_mark2camera = [r.astype(np.float64) for r in R_mark2camera]
    t_mark2camera = [t.astype(np.float64).reshape(3, 1) for t in t_mark2camera]

    return t_mark2camera, R_mark2camera

def compute_grriper2base(pose_list):
    t_gripper2base = []
    R_gripper2base = []
    for i, pose in enumerate(pose_list):
        t_pose = pose[0:3]
        t_pose_ = [t / 1000 for t in t_pose]
        r_pose = pose[3:6]
        gama_z = r_pose[2]
        beta_y = r_pose[1]
        alpha_x = r_pose[0]

        r_pose_ = [alpha_x, beta_y, gama_z]
        r_matrix_ = eulerAngleToRotateMatrix(r_pose_, 'xyz')
        t_gripper2base.append(t_pose_)
        R_gripper2base.append(r_matrix_)

    t_gripper2base = [np.array(t, dtype=np.float64) for t in t_gripper2base]

    return t_gripper2base, R_gripper2base

'''
手眼标定
'''
def calilbrate_eye_in_hand(R_gripper2base, t_gripper2base,R_mark2camera, t_mark2camera,save_path):
    global error_code, error_message

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_mark2camera, t_mark2camera,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T = np.eye(4, dtype=np.float32)

    R_cam2gripper_array = np.array(R_cam2gripper)
    t_cam2gripper_array = np.array(t_cam2gripper)

    T[:3, :3] = R_cam2gripper_array  # 填充旋转部分
    T[:3, 3] = t_cam2gripper_array.flatten()  # 填充平移部分

    print(f'手眼标定结果：\n{T}')
    np.save(save_path, T)
    return T

def clibrate_test(R_gripper2base, t_gripper2base, R_mark2camera, t_mark2camera,T_camera2gripper,image_number):
    global error_code, error_message




def calibrate(user_input):
    global error_code, error_message
    result_data = {}

    try:
        config = json.loads(user_input)
        config = eval(config)
    except Exception as e:
        logging.error(e)
        error_code = 401
        error_message = 'Your input is not a dictionary ! '
        return result_data
    logging.info(f'输入参数：{config}')

    try:
        image_path = config.get("image_path")   # 图像文件夹路径
        image_number = config.get("image_number")   # 图像数量
        pose_file_name = config.get("pose_file_name")   # 机械臂位姿文件名称
        parameter_save_path = config.get("parameter_save_path")     # 参数保存路径
        calibrate_patameter = config.get("calibrate_patameter")     # 手眼标定参数名称
        image_save_path = config.get("image_save_path")     #手眼标定结果图片保存路径
        calibrate_camera = config.get("calibrate_camera")   # 是否进行相机内参标定  1，是；0，否
        camera_parameter = config.get("camera_parameter")   # 相机内参名称
        pattern_size = config.get("pattern_size")  # 标定板规格，类型为字符串如"9 16"，使用空格隔开
        square_size = config.get("square_size")     # 标定板单位尺寸，类型为字符串
        serial_number = config.get("serial_number")

    except Exception as e:
        logging.error(f'[{e}')
        error_code = 402
        error_message = 'Your input is invalid ! '
        return result_data

    pose_file_path = os.path.join(image_path, pose_file_name)
    calibrate_patameter_save_path = os.path.join(parameter_save_path, calibrate_patameter)
    is_calibrate_camera = int(calibrate_camera)

    pattern_size_list = pattern_size.split(' ')
    pattern_size_turple = tuple(int(data) for data in pattern_size_list)
    square_size = float(square_size)

    poses_list = read_robot_poses(pose_file_path)


    if is_calibrate_camera == 1:
        # 进行相机内参标定，路径为参数的保存路径
        camera_parameter_save_path = os.path.join(parameter_save_path, camera_parameter)
        K, distCoeffs = calibrate_camera(image_path, image_number, serial_number, pattern_size_turple, square_size,
                                         image_save_path)

        np.savez(camera_parameter_save_path, K=K, distCoeffs=distCoeffs)

    else:
        # 不进行相机内参标定，路径为读取相机内参路径
        camera_parameter_path = os.path.join(parameter_save_path, camera_parameter)
        K, distCoeffs = load_camera_parameters(camera_parameter_path)

    '''
    image_folder, image_number, K, distCoeffs,board_type='chessboard',
                        pattern_size = (9, 6), square_size = 0.02, save_corner = False, save_path=None'''
    t_mark2camera, R_mark2camera = compute_mark2camera(image_path, image_number, K, distCoeffs, serial_number=serial_number,
                                                       board_type='chessboard', pattern_size = pattern_size_turple,
                                                       square_size = square_size,
                                                       save_corner = True, save_path=image_save_path)
    t_gripper2base, R_gripper2base = compute_grriper2base(poses_list)

    T_camera2gripper = calilbrate_eye_in_hand(R_gripper2base, t_gripper2base,R_mark2camera, t_mark2camera,
                                              calibrate_patameter_save_path)

    logging.info("手眼标定结果测试：")













def main():
    global error_message
    global error_code
    while (True):
        user_input = input("> ")
        if user_input.lower() == 'exit':
            logging.info(f"exit time :{datetime.datetime.now()}")
            return
        try:
            calibrate(user_input)

        except Exception as e:
            logging.error(f"Error: {e}")




if __name__ == '__main__':
  main()