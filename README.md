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

-------
2025年6月15日

更新内容：**calibrate_eye_in_hand_dev.py**

修改**是否需要进行内参标定**，每次标定均进行内参标定。

修改后参数：
* image_path:图像文件夹路径，标定用机械臂坐标文件亦在其中;
* image_number:图像数量;
* pose_file_name:机械臂坐标文件名;
* parameter_save_path:标定参数存放位置，包括相机内参和手眼标定参数;
* image_save_path:标定结果保存路径;
* pattern_size：标定板规格；
* square_size：标定板单位尺寸；
* serial_number：标定序列号，用于区分不同的标定结果;

输出:
* 相机内参保存路径；
* 手眼标定参数保存路径。

D:/SC_MES/Locate2D/Images/CalibrateImages20250613

输入参数示例：
```json
"{\"image_path\":\"D:/SC_MES/Locate2D/Images/CalibrateImages20250613\",\"image_number\":\"19\",\"pose_file_name\":\"poses.txt\",\"parameter_save_path\":\"D:/SC_MES/Locate2D/Parameters\",\"image_save_path\":\"D:/SC_MES/Locate2D/Images/ResultImages/Calibrate\",\"pattern_size\":\"9 6\",\"square_size\":\"0.003\",\"serial_number\":\"20250615\"}"
```
-------
2025年6月18日

更新内容：
* **calibrate_eye_in_hand_dev.py**
* **locate_aruco_dev.py**
* **utils_dev.py**


**calibrate_eye_in_hand_dev.py**
* 添加相对位置计算。

入参变化：
* calibrate_type：标定类型，1表示进行手眼标定，其余参数参考手眼标定参数；2表示进行相对位置计算。
* 标定类型为2时的入参：
  * image_path：计算Mark点位置时的图片；
  * target_pose：目标位姿；
  * robot_pose：机械臂拍照时的位姿；
  * camera_prameter_path：相机内参路径；
  * hand_eye_prameter_path：手眼标定参数路径；
  * comput_type：计算类型，1表示第一阶段参数计算，2表示第二阶段计算；
  * parameter_save_path：参数保存路径。
* 关于计算类型参数：
  * 1：只计算x,y,z,保存为.npy文件
  * 2：同时计算x,y,z,rx,ry,rz，并保存为.npy文件。


**calibrate_eye_in_hand_dev.py**

* 可以计算2次拍照的相对位置

入参变化：
* 添加一个参数**locate_type**：
  * 1：表示第一次定位，此时仅计算第二次拍照位置的x,y,z, 不计算旋转角，但会返回旋转角，请上位机不要采用；
  * 2：表示第二次定位，上位机需要采用x,y,z,rz。

------
更新日期：2025年7月21日
**locate_aruco_dev.py**
更新内容：
添加第一次拍照时的中心点调整功能，使得未居中的Mark点调整到画面中心。
修改：
* 添加一个出参`at_center`，用于向上位机说明此次拍照时的Mark点是否在中心位置，'0'表示不在中心，'1'表示在中心。


