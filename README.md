# Locate2D

项目打包命令：
```shell
pyinstaller --paths .\dev_code --add-data ".\Parameters\target2mark_matrix0521.npy" .\dev_code\locate_aruco.py

pyinstaller .\
```

------
2025年5月22日

代码结构：

* code：用于发布的代码(publish)
  * locate_aruco.py
  * utils.py
  * calibrate_eye_in_hand.py
* dev_code：开发过程的代码
  * locate_aruco_dev.py
  * utils_dev.py
  * calibrate_eye_in_hand_dev.py

------
2025年5月26日

**calibrate_eye_in_hand_dev.py**

接受输入：
* image_path:图像文件夹路径，标定用机械臂坐标文件亦在其中。
* parameter_save_path:标定参数存放位置，包括相机内参和手眼标定参数。
* image_save_path:标定结果保存路径。
* calibrate_camera:是否需要进行相机内参标定，若不需要，则还需提供相机内参文件名称（内参位于parameter_save_path中）
* camera_parameter:相机内参文件名称，若需要进行内参标定，则将标定好的内参保存为该文件名。需要和上述的参数文件路径保持一致。

输出:
* output_perameter_path:相机参数保存路径

------
2025年5月27日

更新内容：**calibrate_eye_in_hand_dev.py**





