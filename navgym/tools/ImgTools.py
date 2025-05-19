import cv2
import numpy as np
import math
from gsamllavanav.mapdata import GROUND_LEVEL
import rasterio
from copy import deepcopy
from gsamllavanav.space import Pose4D, Point2D


def draw_landmarks(image, px_arrays, landmark_names):
    """
    Draw multiple polygon areas on the image and add text labels at the centroid of each area.
    :param image: Image
    :param px_arrays: List of pixel coordinate arrays for multiple polygon areas
    """
    image = np.dstack((image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255))  # BGRA format
    point_data = []
    for px_array in px_arrays:
        # Create a new temporary image for drawing filled polygons
        overlay = np.zeros_like(image)

        # Fill polygon (semi-transparent green, opacity 128)
        cv2.fillPoly(overlay, px_array, color=(255, 0, 0, 80))  # Using BGRA format (BGR+Alpha)

        # Extract Alpha channel
        alpha = overlay[:, :, 3] / 255.0  # Get alpha channel normalized to [0, 1]
        alpha_inv = 1.0 - alpha

        # Composite image: blend transparent layer with original image
        for c in range(3):  # Blend BGR channels (first three channels)
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
    Draw a star marker on the image
    :param image: Image
    :param p: Pixel coordinate (x, y)
    :param color: Color (B, G, R, A)
    """
    cv2.drawMarker(image, p, color, cv2.MARKER_STAR, 10, 2)  # Draw star at the point


def draw_area(image, area):
    """
    Draw a polygon area on the image
    :param image: Image
    :param area: Pixel coordinate array of the polygon area (N, 2)
    """
    area = area.reshape((-1, 1, 2)).astype(np.int32)  # Convert to (N, 1, 2)
    # overlay = np.zeros_like(image)
    # cv2.fillPoly(overlay, [area], color=(255, 255, 0, 100))  # Using BGRA format (BGR+Alpha)
    # alpha = overlay[:, :, 3] / 255.0  # Get alpha channel normalized to [0, 1]
    # alpha_inv = 1.0 - alpha
    # for c in range(3):  # Blend BGR channels
    #     image[:, :, c] = (alpha_inv * image[:, :, c] + alpha * overlay[:, :, c]).astype(np.uint8)
    cv2.polylines(image, [area], isClosed=True, color=(255, 255, 0, 255), thickness=2)  # Draw polygon outline


def draw_triangle(img, center, direction, size=10, color=(0, 0, 255), thickness=2):
    """
    Draw a triangle marker pointing toward the next point
    :param img: Image
    :param center: Center point of the triangle (x, y)
    :param direction: Direction vector (dx, dy)
    :param size: Triangle size
    :param color: Triangle color
    :param thickness: Border thickness
    """
    # Normalize direction vector
    dx, dy = direction
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length == 0:  # Avoid zero vector
        return
    dx /= length
    dy /= length

    # Calculate the three vertices of the triangle
    tip = (int(center[0] + dx * size), int(center[1] + dy * size))  # Tip vertex
    left = (int(center[0] - dy * size / 2), int(center[1] + dx * size / 2))  # Left vertex
    right = (int(center[0] + dy * size / 2), int(center[1] - dx * size / 2))  # Right vertex

    # Draw the triangle
    points = np.array([tip, left, right], np.int32)
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
    cv2.fillPoly(img, [points], color=color)  # Fill the triangle


def draw_arrow(img, endpoint, direction, size=20, color=(0, 0, 255, 255), thickness=10, head_size=0.3):
    """
    Draw an arrow (line plus triangle) on the image
    :param img: Image
    :param endpoint: Arrow endpoint coordinate (x, y)
    :param direction: Direction vector (dx, dy)
    :param size: Overall arrow length
    :param color: Arrow color (B, G, R, A)
    :param thickness: Line thickness
    :param head_size: Triangle head size ratio
    """
    # Normalize direction vector
    dx, dy = direction
    
    # Calculate arrow start point
    start_point = (
        int(endpoint[0] + dx * size),
        int(endpoint[1] + dy * size)
    )
    
    # Draw line
    cv2.line(img, endpoint, start_point, color, thickness)
    
    # Draw triangle at the endpoint
    draw_triangle(img, start_point, (dx, dy), size=int(size * head_size), color=color, thickness=thickness)


def crop_trajectory(image, px_trajectory, area, savefig, directions=None):
    """
    Draw a trajectory on the image and add triangle markers at each point
    :param image: Image
    :param px_trajectory: List of pixel coordinates for the trajectory
    :param area: Pixel coordinate array of the polygon area (N, 2)
    :param savefig: Save path or flag
    :param directions: List of direction vectors
    """
    if len(px_trajectory) > 1:
        cv2.line(image, px_trajectory[-2], px_trajectory[-1], (0, 0, 255, 255), 10)  # Draw line segment
        # Calculate direction vector
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
    Compute the row and column coordinates of the four corners of the field of view area
    :param map_name: Map name
    :param raster: Elevation image
    :param pose: Robot pose
    :return: Row and column coordinates of the four corners of the field of view area
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
    # Variable length
    # 2 * shape_len * |left| => 2 * altitude_from_ground (m) => 1 px = 0.1 m => 20 * altitude_from_ground px
    return [Point2D(x, y) for x, y in view_area_corners_xy]


def crop_rpg(
        image, pose: Pose4D, shape: tuple[int, int], raster: rasterio.DatasetReader, 
        map_name: str, shape_real_size: float, transform: tuple[int, int]= (224, 224)
    ):
    """
    Crop the image to the robot's field of view area
    :param image: Image
    :param pose: Robot pose
    :param shape: Output image size (rows, cols)
    :param raster: Elevation image
    :param map_name: Map name
    :return: Cropped image and the column-row coordinates of the four corners of the field of view area
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
    Crop the image to the robot's field of view area
    :param image: Input image
    :param pose: Robot pose
    :param shape: Output image size (rows, cols)
    :param raster: Elevation map image
    :param map_name: Name of the map
    :param shape_real_size: Real-world size of the cropping area
    :param transform: Size to resize the cropped image to (default (256, 256))
    :return: Cropped image with height adjusted and an extra channel dimension
    """
    # Compute the four corners of the field of view area in row-col coordinates
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, raster, pose, shape_real_size)
    # Flip row-col to col-row for perspective transform
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)
    
    img_row, img_col = shape
    # Define the corners of the output image in col-row order
    img_corners_colrow = [(0, 0), (img_col-1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    
    # Get perspective transform matrix from the view area corners to image corners
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    
    # Warp the input image to the new perspective defined by the robot's view area
    cropped_image = cv2.warpPerspective(image, img_transform, shape)
    
    # Adjust height by subtracting cropped image values from the robot's current height (pose.z)
    cropped_image = pose.z - cropped_image
    
    # Resize the cropped height image to the desired output size
    cropped_image = cv2.resize(cropped_image, transform)
    
    # Add a new axis to make the image have a channel dimension (e.g., (H, W, 1))
    return cropped_image[..., np.newaxis]
