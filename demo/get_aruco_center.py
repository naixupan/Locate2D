'''
获取Mark点中心点坐标，并给出Mark点中心点坐标与图像中心的物理距离
'''

import cv2
import numpy as np
from dev_code.utils_dev import *

'''
工作流：
1、读取图片；
2、计算Mark点中心点像素坐标；
3、计算Mark点中心与图像中心坐标的距离差值；
4、计算机械臂需要移动要的坐标位置
'''

camera_paramter_path = r'D:\GitHubCode\Locate2D\Parameters\camera_params0626_chess.npz'

image_path = r'D:\GitHubCode\Locate2D\Images\ArUcoImages\ArUcoLocate-20250716-7-1.bmp'
image_splite_list = image_path.split('\\')
image_name = image_splite_list[-1].split('.')[0]

print(image_name)

K, distCoeffs = load_camera_parameters(camera_paramter_path)
marker_size = 0.03
save_path =f'../Images/ResultImages/{image_name}.png'
save_path_ = f'../Images/ResultImages/{image_name}_locate.png'

image, corners, _ = detect_aruco_marks(image_path)
# print(corners[0][0][0])

re_image = image.copy()

print(f'图像尺寸:{image.shape}')

image_width = image.shape[1]
image_height = image.shape[0]



for i in range(0, 4):
    point = corners[0][0][i]
    x = int(point[0])
    y = int(point[1])
    cv2.circle(re_image, (x, y), 10, (0, 255, 255), -1, 0, )

center_x = np.mean(corners[0][0][:, 0])
center_y = np.mean(corners[0][0][:, 1])
print(f'({center_x}, {center_y})')

cv2.circle(re_image, (int(center_x), int(center_y)), 10, (255, 0, 0), -1, 0, )
cv2.circle(re_image, (int(image_width/2), int(image_height/2)), 15, (0, 0, 255), -1, 0, )


cv2.imwrite(save_path, re_image)

at_center = at_center(image,corners)
print(at_center)



Matrix_ = compute_aruco_pose(image_path, K, distCoeffs, marker_size, save_image=True, save_path=save_path_)
print(Matrix_)

