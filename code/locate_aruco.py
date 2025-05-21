import json

import cv2
import numpy as np
from utils import *
import logging
import datetime

error_code = 200
error_message = ""

logging.basicConfig(
    filename="locate_aruco.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
    return matrix_

def locate_aruco_marks(user_imput):

    global error_code, error_message
    result_data = {}

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
        image_path = config.get("image_path")
        robot_poses = config.get("robot_pose")
        camera_prameter_path = config.get("camera_prameter_path")
        hand_eye_prameter_path = config.get("hand_eye_prameter_path")
        save_folder = config.get("save_path")

        

    except Exception as e:
        logging.error(f'[{e}')
        error_code = 402
        error_message = 'Your input is invalid ! '
        return result_data
    
    K , distCoeffs = load_camera_parameters(camera_prameter_path)
    
    cam2gripperMatrix = load_calibrate_matrix(hand_eye_prameter_path)
    
    image_path_list = image_path.split('/')
    # print(image_path_list)
    
    # aruco_image_path1 = f'{image_path_list[0]}/ArUcoTest{image_path_list[1]}-{image_path_list[2]}-1.bmp'
    # aruco_image_path2 = f'{image_path_list[0]}/ArUcoTest{image_path_list[1]}-{image_path_list[2]}-2.bmp'
    # aruco_image_path3 = f'{image_path_list[0]}/ArUcoTest{image_path_list[1]}-{image_path_list[2]}-3.bmp'
    
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
    
    for pose in pose_list:
      pose_num = float(pose)
      robot_pose_num.append(pose_num)
    
    gripper2base = Pose2HomogeneousMatrix(robot_pose_num)
    total_pose = [0, 0, 0, 0, 0, 0]
    avg_pose = [0, 0, 0, 0, 0, 0]
    for i in range(len(aruco_image_path)):
      target2camera = compute_aruco_pose(aruco_image_path[i],K, distCoeffs, 0.03, True, save_path)
      target2base = gripper2base @ cam2gripperMatrix @ target2camera
      
      target_pose = HomogeneousMatrix2Pose(target2base)
      for j in range(6):
        total_pose[j] += target_pose[j]
      
    for i in range(6):
      avg_pose[i] = total_pose[i] / 3
      
      
    result_data["code"] = error_code
    result_data["msg"] = error_message
    result_data["x"] = avg_pose[0]
    result_data["y"] = avg_pose[1]
    result_data["z"] = avg_pose[2]
    result_data["rx"] = avg_pose[3]
    result_data["ry"] = avg_pose[4]
    result_data["rz"] = avg_pose[5]
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
          
          error_code = 200
          error_message = ''
        
        print(json.dumps(result_data, ensure_ascii=False, indent=4))
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
        print("end")
        error_code = 200
        error_message = ''
    
    
if __name__ == "__main__":
  main()