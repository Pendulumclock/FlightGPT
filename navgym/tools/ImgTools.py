import cv2
import numpy as np
import math
from gsamllavanav.mapdata import GROUND_LEVEL
import rasterio
from copy import deepcopy
from gsamllavanav.space import Pose4D, Point2D


def draw_landmarks(image, px_arrays, landmark_names):
    """
    在图像上绘制多个多边形区域，并在每个区域的中心点处添加文本标注
    :param image: 图像
    :param px_arrays: 多个多边形区域的像素坐标数组列表
    """
    image = np.dstack((image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255))  # BGRA 格式
    point_data = []
    for px_array in px_arrays:
        # 创建一个新的临时图像，用于绘制填充的多边形
        overlay = np.zeros_like(image)

        # 填充多边形（半透明绿色，透明度为128）
        cv2.fillPoly(overlay, px_array, color=(255, 0, 0, 80))  # 使用 BGRA 格式（BGR+Alpha）

        # 取出 Alpha 通道
        alpha = overlay[:, :, 3] / 255.0  # 获取Alpha通道并归一化到[0, 1]
        alpha_inv = 1.0 - alpha

        # 合成图像：将透明图层与原图合成
        for c in range(3):  # 对BGR通道（前三个通道）进行合成
            image[:, :, c] = (alpha_inv * image[:, :, c] + alpha * overlay[:, :, c]).astype(np.uint8)


        # Calculate centroid using mean of all points
        point_data.append((int(np.mean(px_array[0][:, 0])), int(np.mean(px_array[0][:, 1]))))

    
    for i in range(len(point_data)):
        # Add text annotation at the centroid
        cv2.putText(
            image, landmark_names[i], (point_data[i][0], point_data[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 0, 0, 255), 2
        )
    return image


def draw_star(image, p, color=(255, 255, 0, 255)):
    """
    在图像上绘制一个星形标记
    :param image: 图像
    :param p: 像素坐标 (x, y)
    :param color: 颜色 (B, G, R, A)
    """
    cv2.drawMarker(image, p, color, cv2.MARKER_STAR, 10, 2)  # Draw star at last point


def draw_area(image, area):
    """
    在图像上绘制一个多边形区域
    :param image: 图像
    :param area: 多边形区域的像素坐标数组 (N, 2)
    """
    area = area.reshape((-1, 1, 2)).astype(np.int32)  # 转换为 (4, 1, 2)
    # overlay = np.zeros_like(image)
    # cv2.fillPoly(overlay, [area], color=(255, 255, 0, 100))  # 使用 BGRA 格式（BGR+Alpha）
    # # 取出 Alpha 通道
    # alpha = overlay[:, :, 3] / 255.0  # 获取Alpha通道并归一化到[0, 1]
    # alpha_inv = 1.0 - alpha
    # # 合成图像：将透明图层与原图合成
    # for c in range(3):  # 对BGR通道（前三个通道）进行合成
    #     image[:, :, c] = (alpha_inv * image[:, :, c] + alpha * overlay[:, :, c]).astype(np.uint8)
    cv2.polylines(image, [area], isClosed=True, color=(255, 255, 0, 255), thickness=2)  # Draw polygon


def draw_triangle(img, center, direction, size=10, color=(0, 0, 255), thickness=2):
    """
    绘制一个方向指向下一个点的三角形标记
    :param img: 图像
    :param center: 三角形的中心点 (x, y)
    :param direction: 三角形的方向向量 (dx, dy)
    :param size: 三角形大小
    :param color: 三角形颜色
    :param thickness: 边框厚度
    """
    # 归一化方向向量
    dx, dy = direction
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length == 0:  # 避免零向量
        return
    dx /= length
    dy /= length

    # 计算三角形的三个顶点
    tip = (int(center[0] + dx * size), int(center[1] + dy * size))  # 顶点
    left = (int(center[0] - dy * size / 2), int(center[1] + dx * size / 2))  # 左侧点
    right = (int(center[0] + dy * size / 2), int(center[1] - dx * size / 2))  # 右侧点

    # 绘制三角形
    points = np.array([tip, left, right], np.int32)
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
    cv2.fillPoly(img, [points], color=color)  # 填充三角形


def draw_arrow(img, endpoint, direction, size=20, color=(0, 0, 255, 255), thickness=10, head_size=0.3):
    """
    在图像上绘制一个箭头（直线加三角形）
    :param img: 图像
    :param endpoint: 箭头终点坐标 (x, y)
    :param direction: 方向向量 (dx, dy)
    :param size: 箭头整体长度
    :param color: 箭头颜色 (B, G, R, A)
    :param thickness: 线条粗细
    :param head_size: 三角形头部大小比例
    """
    # 归一化方向向量
    dx, dy = direction
    
    # 计算箭头起点
    start_point = (
        int(endpoint[0] + dx * size),
        int(endpoint[1] + dy * size)
    )
    
    # 绘制直线
    cv2.line(img, endpoint, start_point, color, thickness)
    
    # 在终点绘制三角形
    draw_triangle(img, start_point, (dx, dy), size=int(size * head_size), color=color, thickness=thickness)


def crop_trajectory(image, px_trajectory, area, savefig, directions=None):
    """
    在图像上绘制轨迹，并在每个点处添加三角形标记
    :param image: 图像
    :param px_trajectory: 轨迹的像素坐标列表
    :param area: 多边形区域的像素坐标数组 (N, 2)
    :param savefig: 保存路径
    :parm directions: 方向向量列表
    """
    if len(px_trajectory) > 1:
        cv2.line(image, px_trajectory[-2], px_trajectory[-1], (0, 0, 255, 255), 10)  # Draw line segments
        # 计算方向向量
        direction = (
            px_trajectory[-1][0] - px_trajectory[-2][0], px_trajectory[-1][1] - px_trajectory[-2][1]
        )
        draw_triangle(image, px_trajectory[-2], direction, size=10, color=(255, 255, 255, 255))
    if savefig:
        re_image = deepcopy(image)
        draw_area(re_image, area)
        if directions is not None:
            for p in px_trajectory:
                draw_arrow(img=re_image, endpoint=px_trajectory[-1], size=100, color=(0, 255, 0, 60), direction=directions)
        draw_star(re_image, px_trajectory[-1], color=(0, 255, 0, 255))
        return re_image
    return None


def _compute_view_area_corners_rowcol(map_name: str, raster: rasterio.DatasetReader, pose: Pose4D, shape_real_size:float):
    """
    计算视野区域的四个角点的行列坐标
    :param map_name: 地图名称
    :param raster: 高程图像
    :param pose: 机器人位姿
    :return: 视野区域的四个角点的行列坐标
    """
    view_area_corners_rowcol = [raster.index(x, y) for x, y in view_area_corners(pose, shape_real_size)]
    # view_area_corners_rowcol = [raster.index(x, y) for x, y in view_area_corners(pose, pose.z - GROUND_LEVEL[map_name])]
    return np.array(view_area_corners_rowcol, dtype=np.float32)


def view_area_corners(pose: Pose4D, shape_len: float):
    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    front = np.array([cos, sin])
    left = np.array([-sin, cos])
    center = np.array([pose.x, pose.y])

    view_area_corners_xy = [
        center + shape_len * (front + left),
        center + shape_len * (front - left),  # front right
        center + shape_len * (-front - left),  # back right
        center + shape_len * (-front + left),  # back left
    ]
    # 变长
    # 2 * shape_len * ｜left｜ =》 2* altitude_from_ground （m） =》 1 px = 0.1 m =》 20 * altitude_from_ground px
    return [Point2D(x, y) for x, y in view_area_corners_xy]


def crop_rpg(
        image, pose: Pose4D, shape: tuple[int, int], raster: rasterio.DatasetReader, 
        map_name: str, shape_real_size: float, transform: tuple[int, int]= (224, 224)
    ):
    """
    将图像裁剪为机器人视野区域的图像
    :param image: 图像
    :param pose: 机器人位姿
    :param shape: 输出图像的尺寸 (rows, cols)
    :param raster: 高程图像
    :param map_name: 地图名称
    :return: 裁剪后的图像，视野区域的四个角点的列行坐标
    """
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, raster, pose, shape_real_size)
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
    img_row, img_col = shape
    img_corners_colrow = [(0, 0), (img_col-1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    cropped_image = cv2.warpPerspective(image, img_transform, shape)
    return cv2.resize(cropped_image, transform), view_area_corners_colrow


def crop_height(
        image, pose: Pose4D, shape: tuple[int, int], 
        raster: rasterio.DatasetReader, map_name: str, 
        shape_real_size: float, transform: tuple[int, int]= (256, 256)
    ):
    """
    将图像裁剪为机器人视野区域的图像
    :param image: 图像
    :param pose: 机器人位姿
    :param shape: 输出图像的尺寸 (rows, cols)
    :param raster: 高程图像
    :param map_name: 地图名称
    :return: 裁剪后的图像，视野区域的四个角点的列行坐标
    """
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, raster, pose, shape_real_size)
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
    img_row, img_col = shape
    img_corners_colrow = [(0, 0), (img_col-1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    cropped_image = cv2.warpPerspective(image, img_transform, shape)
    cropped_image = pose.z - cropped_image
    cropped_image = cv2.resize(cropped_image, transform)
    return cropped_image[..., np.newaxis]
