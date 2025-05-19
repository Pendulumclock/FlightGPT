"""
An rgbd image client that uses cropped raster image.

Examples
--------
>>> cropclient.load_image_cache()
>>> rgb, depth = cropclient.crop_view_area('birmingham_block_0', Pose4D(350, 243, 30, np.pi/4), (500, 500))
"""
import gc
from typing import Literal

import cv2
import numpy as np
import math
import rasterio
import rasterio.mask
from copy import deepcopy
from tqdm import tqdm

from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.defaultpaths import ORTHO_IMAGE_DIR
from gsamllavanav.space import Pose4D, view_area_corners


# module-wise image cache
_raster_cache = None
_rgb_cache = None
_height_cache = None  # can be converted to depth


def get_rgbd(map_name: str, pose: Pose4D, rgb_size: tuple[int, int], depth_size: tuple[int, int]):
    
    rgb = crop_image(map_name, pose, rgb_size, 'rgb')
    depth = crop_image(map_name, pose, depth_size, 'depth')

    return rgb, depth

def _crop_image_from_center(image, center_x, center_y, target_width, target_height):
    """
    从指定中心点裁剪固定尺寸的图像
    Args:
        image: 输入图像
        center_x, center_y: 中心点坐标
        target_width, target_height: 目标尺寸
    """
    # 获取原图尺寸
    h, w = image.shape[:2]
    
    # 计算裁剪区域
    x1 = int(center_x - target_width // 2)
    y1 = int(center_y - target_height // 2)
    x2 = x1 + target_width
    y2 = y1 + target_height
    
    # 创建填充后的图像
    padded = np.zeros((target_height, target_width, image.shape[2]) if len(image.shape) == 3 else (target_height, target_width), dtype=image.dtype)
    
    # 计算实际裁剪和填充区域
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    
    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)
    dst_x2 = target_width - max(0, x2 - w)
    dst_y2 = target_height - max(0, y2 - h)
    
    # 复制有效区域
    padded[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    return padded


def _get_px_list(map_name: str, pose_list: list) -> list[tuple[int, int]]:
    return [_get_pose_px(map_name, pose) for pose in pose_list]

def _get_pose_px(map_name: str, pose: Pose4D) -> tuple[int, int]:
    # view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
    # # 计算中心点 (平均值)
    # center = np.mean(view_area_corners_rowcol, axis=0)
    
    # # 转换为整数坐标
    # center_x, center_y = int(center[1]), int(center[0])  # row,col -> x,y
    # return [center_x, center_y]
    raster = _raster_cache[map_name]

    center_x, center_y = raster.index(pose.x, pose.y)
    return center_y, center_x

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


def crop_image(map_name: str, pose: Pose4D, shape: tuple[int, int], type: Literal['rgb', 'depth', 'trajectory'], trajectory=None, px_arrays=None, landmark_names=None) -> np.ndarray:
    if type == 'trajectory':
        # 假设 cropclient._rgb_cache[eps.map_name] 是原始RGB图像
        image = deepcopy(_rgb_cache[map_name])

        # 将RGB图像与Alpha通道合并，形成RGBA图像
        image = np.dstack((image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255))  # BGRA 格式
        for px_array, landmark_name in zip(px_arrays, landmark_names):
            # 创建一个新的临时图像，用于绘制填充的多边形
            overlay = np.zeros_like(image)

            # 填充多边形（半透明绿色，透明度为128）
            cv2.fillPoly(overlay, px_array, color=(0, 255, 255, 80))  # 使用 BGRA 格式（BGR+Alpha）

            # 取出 Alpha 通道
            alpha = overlay[:, :, 3] / 255.0  # 获取Alpha通道并归一化到[0, 1]
            alpha_inv = 1.0 - alpha

            # 合成图像：将透明图层与原图合成
            for c in range(3):  # 对BGR通道（前三个通道）进行合成
                image[:, :, c] = (alpha_inv * image[:, :, c] + alpha * overlay[:, :, c]).astype(np.uint8)


            # Calculate centroid using mean of all points
            centroid_x = int(np.mean(px_array[0][:, 0]))
            centroid_y = int(np.mean(px_array[0][:, 1]))

            # Add text annotation at the centroid
            cv2.putText(image, landmark_name, 
                        (centroid_x, centroid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0, 255), 2)
        p = []
        for t in trajectory:
            p.append(_get_pose_px(map_name, t))
        # Draw the trajectory line with stars
        for i in range(len(p)-1):
            cv2.line(image, p[i], p[i+1], (255, 0, 0, 255), 10)  # Draw line segments
            # 计算方向向量
            direction = (p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
            # 绘制三角形方向标记
            draw_triangle(image, p[i], direction, size=10, color=(255, 255, 255, 255))
            # cv2.drawMarker(image, p[i], (0, 0, 255), cv2.MARKER_STAR, 10, 2)  # Draw stars at points
        cv2.drawMarker(image, p[-1], (0, 255, 0, 255), cv2.MARKER_STAR, 10, 2)  # Draw star at last point
        cv2.drawMarker(image, p[0], (255, 255, 0, 255), cv2.MARKER_STAR, 10, 2)  # Draw star at last point
        return image # _crop_image_from_center(image, p[-1][0], p[-1][1], shape[0], shape[1])

    image = (_rgb_cache if type =='rgb' else _height_cache)[map_name]
    # 算四个角点 - 剪下来 - 透视变幻 - 压缩到指定大小
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)  # 计算视野区域的四个角点
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

    img_row, img_col = shape
    img_corners_colrow = [(0, 0), (img_col-1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    cropped_image = cv2.warpPerspective(image, img_transform, shape)

    if type == 'depth':
        cropped_image = pose.z - cropped_image
        cropped_image = cropped_image[..., np.newaxis]

    return cropped_image


def _compute_view_area_corners_rowcol(map_name: str, pose: Pose4D):
    """Returns the [front-left, front-right, back-right, back-left] corners of
    the view area in (row, col) order
    """

    raster = _raster_cache[map_name]

    view_area_corners_rowcol = [raster.index(x, y) for x, y in view_area_corners(pose, GROUND_LEVEL[map_name])]

    return np.array(view_area_corners_rowcol, dtype=np.float32)


def load_image_cache(image_dir=ORTHO_IMAGE_DIR, alt_env: Literal['', 'flood', 'ground_fissure'] = ''):
    if alt_env:
        image_dir = image_dir/alt_env

    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is None:
        _raster_cache = {
            raster_path.stem: rasterio.open(raster_path)
            for raster_path in image_dir.glob("*.tif")
        }

    if _rgb_cache is None:
        _rgb_cache = {
            rgb_path.stem: cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            for rgb_path in tqdm(image_dir.glob("*.png"), desc="reading rgb data from disk", leave=False)
        }

    if _height_cache is None:
        _height_cache = {
            map_name: raster.read(1)  # read first channel (1-based index)
            for map_name, raster in tqdm(_raster_cache.items(), desc="reading depth data from disk", leave=False)
        }



def clear_image_cache():
    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is not None:
        for dataset in _raster_cache:
            dataset.close()
    
    _raster_cache = _rgb_cache = _height_cache = None
    gc.collect()
