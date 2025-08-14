import numpy as np
camera_paramter_path = '../Parameters/camera_parameter_20250615_chess.npz'
camera_parameter= np.load(camera_paramter_path)
K = camera_parameter['K']
distCoeffs = camera_parameter['distCoeffs']
print(f'相机内参：{K}')
print(f'畸变系数：{distCoeffs}')