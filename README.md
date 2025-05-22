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

