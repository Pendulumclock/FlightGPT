from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.maps.gsam_map import GSamParams
import cv2
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories_by_path
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.cityreferobject import get_city_refer_objects
import rasterio
from pathlib import Path
import rasterio.mask
from gsamllavanav.dataset.episode import Episode
from tqdm import tqdm
import numpy as np
from copy import deepcopy


class SingleCityNavData:
    def __init__(
            self, episode: Episode, map: LandmarkNavMap, rgb: np.ndarray, height: np.ndarray, 
            px_list: list[tuple[int, int]], raster: rasterio.DatasetReader):
        """
        Used to store single city navigation data
        Args:
            episode: episode, target etc
            map: map, 5 types of map data
            rgb: rgb data, RGB image
            height: height data, height image
            px_list: px_list, landmark edge coordinates
            raster: raster grid (used for querying coordinates to px)
        """
        self.episode = deepcopy(episode)
        self.map = deepcopy(map)
        self.rgb = deepcopy(rgb)
        self.height = deepcopy(height)
        self.px_list = px_list
        self.raster = raster


class CityNavData:
    def __init__(
            self, path, fix_altitude=50, max_dist_marker_to_target=1000000, 
            map_shape=(240, 240), map_pixels_per_meter=240/410, use_segmentation_mask=True,
            use_bbox_confidence=False, box_threshold=0.2, text_threshold=0.25,
            max_box_size=50, max_box_area=3000, image_dir="./data/rgbd-new"
        ):
        """
        Used to store city navigation data
        Args:
            path: path, mturk data path
            fix_altitude: fix_altitude, altitude correction
            max_dist_marker_to_target: max_dist_marker_to_target, controls how many trajectories are imported, 1000000 means import all
            map_shape: map_shape, the size to which the 5 maps are downscaled for visualization
            map_pixels_per_meter: map_pixels_per_meter, pixels per meter on the map
            use_segmentation_mask: use_segmentation_mask, whether to use segmentation mask
            use_bbox_confidence: use_bbox_confidence, whether to use bbox confidence
            box_threshold: box_threshold, bbox threshold
            text_threshold: text_threshold, text threshold
            max_box_size: max_box_size, maximum bbox size
            max_box_area: max_box_area, maximum bbox area
            image_dir: image_dir, image path
        """
        # Load cityrefer object.json
        objects = get_city_refer_objects()
        image_dir = Path(image_dir)
        # Load mturk data, data/citynav/citynav_{split}_{difficulty}.json, 'all' means all three difficulties
        murk = load_mturk_trajectories_by_path(path=path, fix_altitude=fix_altitude)
        # Generate episodes, list (max_dist_marker_to_target controls how many trajectories are imported, 1000000 means all)
        self.episodes = generate_episodes_from_mturk_trajectories(objects, murk, max_dist_marker_to_target=max_dist_marker_to_target)
        # Generate maps based on episodes
        # gsam config parameters
        gp = GSamParams(
            use_segmentation_mask=use_segmentation_mask,
            use_bbox_confidence=use_bbox_confidence,
            box_threshold=box_threshold, text_threshold=text_threshold,
            max_box_size=max_box_size, max_box_area=max_box_area
        )
        self.maps = [
            LandmarkNavMap(
                map_name=eps.map_name, map_shape=map_shape, map_pixels_per_meter=map_pixels_per_meter,
                landmark_names=eps.description_landmarks, target_name=eps.description_target, 
                surroundings_names=eps.description_surroundings, gsam_params=gp
            ) for eps in self.episodes
        ]

        self.data_len = len(self.episodes)

        self._raster_cache = {
            raster_path.stem: rasterio.open(raster_path)
            for raster_path in image_dir.glob("*.tif")
        }

        self._rgb_cache = {
            rgb_path.stem: cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            for rgb_path in tqdm(image_dir.glob("*.png"), desc="reading rgb data from disk", leave=False)
        }

        self._height_cache = {
            map_name: raster.read(1)  # read first channel (1-based index)
            for map_name, raster in tqdm(self._raster_cache.items(), desc="reading depth data from disk", leave=False)
        }
    
    def _get_pose_px(self, pose, map_name):
        """
        Get the pixel coordinates of a pose
        Args:
            pose: pose, position
            map_name: map_name, map name under rgb
        Returns:
            [int(center_y), int(center_x)]: pixel coordinates of the pose
        """
        raster = self._raster_cache[map_name]
        center_x, center_y = raster.index(pose.x, pose.y)
        return [int(center_y), int(center_x)]
    
    def _get_px_list(self, map_name, contour):
        """
        Get the pixel list
        Args:
            map_name: map_name, map name
            contour: contour, contour
        Returns:
            list of pixel coordinates for the contour
        """
        return [self._get_pose_px(pose, map_name) for pose in contour]
    
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.data_len
    
    def __getitem__(self, idx):
        """
        Get data by index
        Args:
            idx: idx, index
        Returns:
            SingleCityNavData(
                episode=self.episodes[idx], map=self.maps[idx], 
                rgb=self._rgb_cache[self.episodes[idx].map_name], 
                height=self._height_cache[self.episodes[idx].map_name],
                px_list=px_list, raster=self._raster_cache[self.episodes[idx].map_name]
            ) single city navigation data
        """
        px_list = [
            np.array([self._get_px_list(self.episodes[idx].map_name, x.contour)], dtype=np.int32)
            for x in self.maps[idx].landmark_map.landmarks
        ]
        return SingleCityNavData(
            episode=self.episodes[idx], map=self.maps[idx], 
            rgb=deepcopy(self._rgb_cache[self.episodes[idx].map_name]), 
            height=deepcopy(self._height_cache[self.episodes[idx].map_name]),
            px_list=px_list, raster=self._raster_cache[self.episodes[idx].map_name]
        )
