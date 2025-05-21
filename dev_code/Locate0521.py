'''
文件名称：CalibrateTest0428
作者：何广鹏
日期：2025年4月28日
功能：手眼标定测试
版本：V1.2
'''

'''
手眼标定测试流程：
1、加载手眼标定结果和相机内参
2、检测ArUco标记，获得ArUco标记在相机下的坐标
3、根据机械臂位姿计算ArUco标记在世界坐标中的位姿
4、将标定板位姿矩阵转化为x,y,z,rx,ry,rz
'''

import cv2
import numpy as np
from utils_dev import *

def compute_aruco_pose(image_path,K,distCoeffs,marker_size, save_image=True):

    image, corners, ids = detect_aruco_marks(image_path)
    if ids is not None:
        rvec, tvec, mark_in_camera, camera_in_mark = compute_camera_pose(corners, ids, K, distCoeffs,marker_size)
        R, _ = cv2.Rodrigues(rvec)
        print(f"相机平移向量：{mark_in_camera}")
        print(f"相机位置：{camera_in_mark}")
        if save_image:
            visualize_pose(image, rvec, tvec, corners, ids, mark_in_camera, camera_in_mark, K, distCoeffs)
            R_arr = np.array(R, dtype=np.float64)
    t_arr = np.array(tvec, dtype=np.float64).flatten()
    # t_arr_ = t_arr * 1000

    matrix_ = np.eye(4, dtype=np.float32)
    matrix_[:3, :3] = R_arr
    matrix_[:3, 3] = t_arr
    return matrix_






if __name__ == '__main__':
    camera_parameter_path = './Parameters/camera_params0513_chess.npz'
    calibrate_matrix_path = './Parameters/cam2gripper_matrix0513_chess.npy'
    target2mark_matrix_path = './Parameters/target2mark_matrix0521.npy'

    date = '0513'
    count = '36'

    aruco_image_path = [f'./Images/test/ArUcoTest{date}-{count}-1.bmp',f'./Images/test/ArUcoTest{date}-{count}-2.bmp', f'./Images/test/ArUcoTest{date}-{count}-3.bmp']

    K , distCoeffs = load_camera_parameters(camera_parameter_path)
    cam2gripper = load_calibrate_matrix(calibrate_matrix_path)
    T_target2mark = load_calibrate_matrix(target2mark_matrix_path)

    robot_pose = [47.4705, -720.6140, -111.1109, 179.6020, -1.4580, -89.3855]
    gripper2base = Pose2HomogeneousMatrix(robot_pose)

    toltal_pose = [0,0,0,0,0,0]
    for i in range(3):
        mark2camera = compute_aruco_pose(aruco_image_path[i],K,distCoeffs,0.03,True)
        mark2base = gripper2base @ cam2gripper @ mark2camera
        mark_pose = HomogeneousMatrix2Pose(mark2base)
        print(f"目标位姿[{i}]：{mark_pose}")
        for j in range(6):
            toltal_pose[j] += mark_pose[j]

    for i in range(6):
        toltal_pose[i] = toltal_pose[i]/3
    print(f"平均目标位姿：{toltal_pose}")
    
    T_mark2base = Pose2HomogeneousMatrix(toltal_pose)
    T_target2base = T_mark2base @ T_target2mark
    target_pose = HomogeneousMatrix2Pose(T_target2base)
    print(f'偏置位姿：{target_pose}')

