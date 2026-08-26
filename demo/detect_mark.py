import cv2


def detect_aruco_marks(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, img = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY);

    cv2.imwrite('../Images/ResultImages/threshold.jpg', img)

    if image is None:
        raise FileNotFoundError(f"未找到图像：{image_path}")
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    return img, corners, ids


if __name__ == '__main__':
    image_path = r'D:\Desktop\test\20250819\ArUcoLocate-20250819-22-2.bmp'
    image,corners,ids = detect_aruco_marks(image_path)
    if ids is not None:
        print("识别到Mark点")
        print(corners)
    else:
        print("未识别到Mark点！")