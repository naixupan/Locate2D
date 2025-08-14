'''
文件名称：utils_dev.py
作者：何广鹏
日期：2025年4月3日
功能：工具函数
版本：V1.0

日期：2025年7月22日
版本：V1.1


'''
import os
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import math


import pandas as pd
from datetime import datetime
import time
import os
# from DetectAruco import detect_aruco_marks, compute_camera_pose, visualize_pose


COLUMNS = [
    'x_compute',
    'y_compute',
    'z_compute',
    'rz_compute',
    'x_original',
    'y_original',
    'z_original',
    'rz_original',
    'x_robot',
    'y_robot',
    'z_robot',
    'rx_robot',
    'ry_robot',
    'rz_robot',
    'type',
    'locate_type'
]

# 加载相机内参
def load_camera_parameters(camera_parameters_path):
    calib_data = np.load(camera_parameters_path)
    K = calib_data['K']
    distCoeffs = calib_data['distCoeffs']
    return K, distCoeffs


# 加载手眼标定结果
def load_calibrate_matrix(matrix_path):
    return np.load(matrix_path)


# 欧拉角转旋转矩阵
def euler_to_rotation_matrix(euler_angles, rotation_order='zyx', degrees=True):
    """
    欧拉角 → 旋转矩阵
    :param euler_angles: 欧拉角（例如 [gamma_z, beta_y, alpha_x]）
    :param rotation_order: 旋转顺序（例如 'zyx', 'xyz'）
    :param degrees: 输入是否为角度（True为角度，False为弧度）
    :return: 3x3旋转矩阵
    """
    rotation = R.from_euler(rotation_order, euler_angles, degrees=degrees)
    R_matrix = rotation.as_matrix()
    return R_matrix


# 旋转矩阵转欧拉角
def rotation_matrix_to_euler(R_matrix, rotation_order='zyx', degrees=True):
    """
    旋转矩阵 → 欧拉角
    :param R_matrix: 3x3旋转矩阵
    :param rotation_order: 旋转顺序（例如 'zyx', 'xyz'）
    :param degrees: 是否返回角度（True返回角度，False返回弧度）
    :return: 欧拉角（数组形式，例如 [gamma_z, beta_y, alpha_x]）
    """
    # 使用Scipy处理万向节锁
    rotation = R.from_matrix(R_matrix)
    euler = rotation.as_euler(rotation_order, degrees=degrees)
    return euler


# 读取机械臂坐标
def read_robot_poses(poses_path):
    fileHandle = open(poses_path, 'r')
    lines = fileHandle.readlines()
    posesList = []
    for i, line in enumerate(lines):
        # print(line)
        split_value = line.split(' ')
        pose_data = split_value[1:]
        pose_data[-1] = pose_data[-1].strip()
        # print(pose_data)
        float_pose = [float(x) for x in pose_data]
        count = i + 1
        float_pose.append(count)
        # print(float_pose)
        posesList.append(float_pose)
    # print(f'posesList: {posesList}')
    return posesList


def eulerAngleToRotateMatrix(eulerAngle, seq):
    rx = math.radians(eulerAngle[0])
    ry = math.radians(eulerAngle[1])
    rz = math.radians(eulerAngle[2])

    rx_sin = np.sin(rx)
    rx_cos = np.cos(rx)
    ry_sin = np.sin(ry)
    ry_cos = np.cos(ry)
    rz_sin = np.sin(rz)
    rz_cos = np.cos(rz)

    RotX = np.array([[1, 0, 0],
                     [0, rx_cos, -rx_sin],
                     [0, rx_sin, rx_cos]], dtype=np.float64)
    RotY = np.array([[ry_cos, 0, ry_sin],
                     [0, 1, 0],
                     [-ry_sin, 0, ry_cos]], dtype=np.float64)
    RotZ = np.array([[rz_cos, -rz_sin, 0],
                     [rz_sin, rz_cos, 0],
                     [0, 0, 1]], dtype=np.float64)

    if seq == 'xyz':
        R_matrix = RotZ @ RotY @ RotX

    return R_matrix


def rotateMatrixToEulerAngle(R_matrix, seq='xyz'):
    """
    将旋转矩阵转换为欧拉角（弧度）
    :param R_matrix: 3x3 旋转矩阵
    :param seq: 欧拉角顺序（仅支持 'xyz'）
    :return: 欧拉角数组 [rx_deg, ry_deg, rz_deg]（单位：度）
    """
    assert R_matrix.shape == (3, 3), "输入必须是3x3矩阵"
    assert seq == 'xyz', "当前仅支持 'xyz' 顺序"

    # 提取旋转矩阵元素
    m00 = R_matrix[0, 0]
    m01 = R_matrix[0, 1]
    m02 = R_matrix[0, 2]
    m10 = R_matrix[1, 0]
    m11 = R_matrix[1, 1]
    m12 = R_matrix[1, 2]
    m20 = R_matrix[2, 0]
    m21 = R_matrix[2, 1]
    m22 = R_matrix[2, 2]

    # 计算Y轴旋转角度（弧度）
    ry = np.arcsin(-m20)
    ry = np.clip(ry, -np.pi / 2, np.pi / 2)  # 限制在 -90° ~ 90° 之间

    # 处理万向锁（当cos(ry)接近0时）
    if abs(np.cos(ry)) < 1e-6:
        # 此时Y轴旋转为 ±90°，X和Z轴无法唯一确定
        rx = 0.0
        rz = np.arctan2(-m01, m11)
    else:
        # 计算X和Z轴旋转角度
        rx = np.arctan2(m21 / np.cos(ry), m22 / np.cos(ry))
        rz = np.arctan2(m10 / np.cos(ry), m00 / np.cos(ry))

    # 弧度转角度
    rx_deg = np.degrees(rx)
    ry_deg = np.degrees(ry)
    rz_deg = np.degrees(rz)
    return [rx_deg, ry_deg, rz_deg]


def R_T2HomogeneousMatrix(R_matrix, T_vector):
    T = np.eye(4, 4, dtype=np.float64)
    T[:3, :3] = R_matrix
    T[:3, 3] = T_vector.flatten()
    return T


def HomogeneousMatrix2R_T(T_matrix):
    R_matrix = T_matrix[:3, :3]
    T_vector = T_matrix[:3, 3]
    return R_matrix, T_vector


def Pose2HomogeneousMatrix(pose):
    t_pose = pose[0:3]
    t_pose_ = [t / 1000 for t in t_pose]
    r_pose = pose[3:6]

    gama_z = r_pose[2]
    beta_y = r_pose[1]
    alpha_x = r_pose[0]

    r_pose_ = [alpha_x, beta_y, gama_z]
    r_matrix_ = eulerAngleToRotateMatrix(r_pose_, 'xyz')

    t_pose_ = np.array(t_pose_, dtype=np.float64).reshape(3, 1)

    T_matrix = R_T2HomogeneousMatrix(r_matrix_, t_pose_)
    return T_matrix


def HomogeneousMatrix2Pose(T_matrix):
    r_matrix, t_vector = HomogeneousMatrix2R_T(T_matrix)
    r_pose = rotation_matrix_to_euler(r_matrix, rotation_order='xyz', degrees=True)
    t_pose = [t * 1000 for t in t_vector.flatten()]
    pose = [t_pose[0], t_pose[1], t_pose[2], r_pose[0], r_pose[1], r_pose[2]]
    return pose


# 检测棋盘格图片，返回棋盘格角点坐标，用于手眼标定
def DetectChessBoard(image_path, pattern_size=(9, 6), square_size=0.02, save_corners=False, save_path=None):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if save_corners:
            cv2.drawChessboardCorners(image, pattern_size, corners_refined, ret)
            cv2.imwrite(save_path, image)
        return objp, corners, ret
    else:
        return None, None, ret


# 检测圆形标定板图片，返回圆形标定板角点坐标，用于手眼标定
def DetectCircleBoard(image_path, pattern_size=(7, 7), square_size=0.004, save_corners=False, save_path=None):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.maxArea = 20000
    params.minArea = 20
    params.minDistBetweenBlobs = 1
    params.filterByCircularity = True  # 启用圆度过滤
    params.minCircularity = 0.6  # 最小圆度阈值（0-1）
    params.maxCircularity = 1.0  # 最大圆度阈值（0-1）
    blobDetector = cv2.SimpleBlobDetector_create(params)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret, corner = cv2.findCirclesGrid(gray, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector, None)
    if ret:
        # corner_refined = refine_circle_centers(gray, corner)
        # corner_refined = cv2.cornerSubPix(gray, corner, pattern_size, (-1, -1), criteria)
        if save_corners:
            img = cv2.drawChessboardCorners(image, pattern_size, corner, ret)
            cv2.imwrite(save_path, img)
        return objp, corner, ret
    else:
        print(f"图像 {image_path} 未检测到足够角点！跳过...")
        return None, None, ret


def refine_circle_centers(gray_image, centers, win_size=15):
    """
    基于灰度质心的亚像素优化（适用于圆点标定板）
    :param gray_image: 单通道灰度图
    :param centers: 初始圆心坐标（像素级，形状为 Nx1x2）
    :param win_size: 局部窗口大小（奇数）
    :return: 优化后的圆心坐标（亚像素级）
    """
    refined_centers = []
    half_win = win_size // 2

    for (x, y) in centers.squeeze():
        x, y = float(x), float(y)
        # 提取局部窗口
        x_min = int(x - half_win)
        x_max = int(x + half_win + 1)
        y_min = int(y - half_win)
        y_max = int(y + half_win + 1)

        # 边界检查
        if x_min < 0 or y_min < 0 or x_max > gray_image.shape[1] or y_max > gray_image.shape[0]:
            refined_centers.append([x, y])
            continue

        patch = gray_image[y_min:y_max, x_min:x_max]
        gy, gx = np.mgrid[-half_win:half_win + 1, -half_win:half_win + 1]

        # 计算质心
        total = np.sum(patch)
        dx = np.sum(gx * patch) / total
        dy = np.sum(gy * patch) / total
        refined_centers.append([x + dx, y + dy])

    return np.array(refined_centers).reshape(-1, 1, 2)


def draw_circle_grid(image, centers, pattern_size=(7, 7)):
    img = image.copy()
    centers_reshaped = centers.reshape(pattern_size[0], pattern_size[1], 2)
    # 绘制圆心和网格连线
    for i in range(pattern_size[0]):
        for j in range(pattern_size[1]):
            pt = tuple(np.round(centers_reshaped[i, j]).astype(int))
            cv2.circle(img, pt, 5, (0, 255, 0), -1)
            if j < pattern_size[1] - 1:
                pt_next = tuple(np.round(centers_reshaped[i, j + 1]).astype(int))
                cv2.line(img, pt, pt_next, (255, 0, 0), 2)
            if i < pattern_size[0] - 1:
                pt_down = tuple(np.round(centers_reshaped[i + 1, j]).astype(int))
                cv2.line(img, pt, pt_down, (255, 0, 0), 2)
    return img


def detect_aruco_marks(image_path):
  image = cv2.imread(image_path)
  
  if image is None:
    raise FileNotFoundError(f"未找到图像：{image_path}")
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
  corners, ids, _ = detector.detectMarkers(image)
  return image, corners, ids
  
def compute_camera_pose(corners, ids, K, distCoeffs, MARKER_SIZE):
  if ids is None:
    return None, None, None, None
  
  rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, K, distCoeffs)
  
  rvec = rvecs[0]
  tvec = tvecs[0].reshape(3, 1)
  
  R, _ = cv2.Rodrigues(rvec)
  camera_in_mark = -np.dot(R.T, tvec).flatten()
  
  # print(f"相机在标记坐标系中的旋转矩阵：{R}")
  
  return rvec, tvec, tvec.flatten(), camera_in_mark
  
def visualize_pose(image, rvec, tvec, corners, ids, marke_in_camera, camera_in_mark, K, distCoeffs,save_path=None):
  # 获取当前时间戳
  # current_datetime = datetime.datetime.now()
  # formatted_datetime = current_datetime.strftime('%Y-%m-%d-%H%M%S')
  image_with_axes = image.copy()
  cv2.drawFrameAxes(image_with_axes, K, distCoeffs, rvec, tvec, 0.1)
  # save_path = os.path.join(SAVE_FOLDER, f"camera_pose_visualize{formatted_datetime}.png")
  cv2.imwrite(save_path, image_with_axes)
  
  # figure = plt.figure(figsize=(12, 6))
  #
  # ax1 = figure.add_subplot(121, projection="3d")
  # ax2 = figure.add_subplot(122, projection="3d")
  #
  # ax1.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
  # ax1.scatter(marke_in_camera[0], marke_in_camera[1], marke_in_camera[2], s=100, c='purple', label='Mark')
  # ax1.set_title("Mark in Camera Coordinate")
  # ax1.legend()
  #
  # ax2.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
  # ax2.scatter(camera_in_mark[0], camera_in_mark[1], camera_in_mark[2], s=100, c='blue', label='Camera')
  # ax2.set_title("Camera in Mark Coordinate")
  # ax2.legend()
  #
  # plt_image_path = os.path.join(SAVE_FOLDER, f"3D_coordinate{formatted_datetime}.png")
  # plt.savefig(plt_image_path, dpi=600, bbox_inches='tight')
  # plt.close()


def compute_target2mark_marix(T_mark2base, T_target2base):
    return np.linalg.inv(T_mark2base) @ T_target2base

def compute_aruco_pose(image_path,K,distCoeffs,marker_size, save_image=True, save_path = None):

    image, corners, ids = detect_aruco_marks(image_path)
    if ids is not None:
        rvec, tvec, mark_in_camera, camera_in_mark = compute_camera_pose(corners, ids, K, distCoeffs,marker_size)
        R, _ = cv2.Rodrigues(rvec)
        # print(f"相机平移向量：{mark_in_camera}")
        # print(f"相机位置：{camera_in_mark}")
        if save_image:
            visualize_pose(image, rvec, tvec, corners, ids, mark_in_camera, camera_in_mark, K, distCoeffs,save_path)
        R_arr = np.array(R, dtype=np.float64)
    t_arr = np.array(tvec, dtype=np.float64).flatten()
    matrix_ = np.eye(4, dtype=np.float32)
    matrix_[:3, :3] = R_arr
    matrix_[:3, 3] = t_arr
    return matrix_ ,corners

def at_center(image, corners):
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Mark点中心点坐标
    mark_center_x = np.mean(corners[0][0][:, 0])
    mark_center_y = np.mean(corners[0][0][:, 1])

    at_center = False

    if mark_center_x > image_width // 3 and mark_center_x < (image_width // 3) * 2:
        if mark_center_y > image_height // 3 and mark_center_y < (image_height // 3) * 2:
            at_center = True

    return at_center

def initalize_csv(file_path,headers):
    if not os.path.exists(file_path):
        pd.DataFrame(columns=headers).to_csv(file_path,index=False)
        # print(f'已创建CSV文件：{file_path}')

        return 0

    else:
        # print(f'CSV文件已存在：{file_path}')
        return 1


def append_to_csv(file_path, data):
    existing_df = pd.read_csv(file_path)
    if isinstance(data,dict):
        new_df = pd.DataFrame([data])
    elif isinstance(data,list) and all(isinstance(item, dict) for item in data):
        new_df = pd.DataFrame(data)
    else:
        raise ValueError("数据格式错误，请提供字典或字典列表")
    for col in COLUMNS:
        if col not in new_df.columns:
            new_df[col] = np.nan

        # 重新排序列以匹配表头
    new_df = new_df[COLUMNS]

    # 合并新旧数据
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 保存回CSV文件
    combined_df.to_csv(file_path, index=False)
    # print(f"已追加 {len(new_df)} 条数据到 {file_path}")
    return True