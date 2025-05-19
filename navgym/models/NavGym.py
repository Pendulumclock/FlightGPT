from navgym.models.CityNavData import SingleCityNavData
from copy import deepcopy
from navgym.tools.ImgTools import draw_landmarks, draw_star, crop_height, crop_rpg, crop_trajectory
from navgym.tools.TimeTools import time_str
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D
import os
import math
from pydantic import BaseModel
import matplotlib.pyplot as plt


action_dict = {
    'Move Forward' : 1, 
    'Turn Left' : 3,
    'Turn Right' : 2,
    'Stop': 0
} # 3 -> + 2-> -
action_list = ['Stop', 'Move Forward', 'Turn Right', 'Turn Left']


class PhotoDirs(BaseModel):
    """
    Format paths
    Args:
        map_photo_path: Map photo path
        rgb_drone_path: RGB drone image path
        height_drone_path: Height (depth) drone image path
        map_current_view_area: Current view area map
        map_explored_area: Explored area map
        map_landmark_map: Landmark map
        map_target_map: Target map
        map_surroundings_map: Surroundings map
    """
    map_photo_path: str
    rgb_drone_path: str
    height_drone_path: str
    map_current_view_area: str
    map_explored_area: str
    map_landmark_map: str
    map_target_map: str
    map_surroundings_map: str


class NavGym:
    def __init__(self, citynav_data: SingleCityNavData, data_dir = os.path.abspath('./R1PhotoData')):
        """
        Store navigation data
        Args:
            citynav_data: citynav_data, single city navigation data
        """
        self.data = citynav_data
        px_names = [x.name for x in self.map.landmark_map.landmarks]
        self.map_photo_with_landmark = deepcopy(self.rgb)
        self.father_image_dir = f'{data_dir}/{self.episode.map_name}_{time_str()}'
        self.cur_pos = self.start_pose
        self.rgb_crop = None
        self.height_crop = None
        self.cur_step = 0
        self.trajectory = []
        drone_view_shape = (self.start_pose.z - GROUND_LEVEL[self.map_name]) / self.px_real_size[0]
        self.drone_view_size_in_meter = drone_view_shape * self.px_real_size[0] * 2
        self.sight_size = (drone_view_shape, drone_view_shape)
        self.drone_view_shape = (int(drone_view_shape), int(drone_view_shape))
        self.drone_view_px_size = drone_view_shape * self.px_real_size[0]
        self.px_trajectory = []
        self.photo_dirs = [] 
        self.actions = []
        self._init_photo(px_names)
    
    def _init_photo(self, px_names):
        """
        Initialize photo
        Args:
            px_names: px_names, landmark names
        """
        self.map_photo_with_landmark = draw_landmarks(self.map_photo_with_landmark, self.px_list, px_names)
        draw_star(self.map_photo_with_landmark, self._get_px(self.start_pose))
        os.mkdir(self.father_image_dir)
        self._generate_photo()
        # print(self.photo_dirs[-1].map_photo_path)
        
    def _get_photo_dirs(self):
        """
        Generate photo paths
        """
        t = time_str()
        self.photo_dirs.append(
            PhotoDirs(
                map_photo_path=f'{self.father_image_dir}/map_{self.cur_step}_{t}.jpg',
                rgb_drone_path=f'{self.father_image_dir}/rgb_drone_{self.cur_step}_{t}.jpg',
                height_drone_path=f'{self.father_image_dir}/height_drone_{self.cur_step}_{t}.jpg',
                map_current_view_area=f'{self.father_image_dir}/map_current_view_area_{self.cur_step}_{t}.jpg',
                map_explored_area=f'{self.father_image_dir}/map_explored_area_{self.cur_step}_{t}.jpg',
                map_landmark_map=f'{self.father_image_dir}/map_landmark_map_{self.cur_step}_{t}.jpg',
                map_target_map=f'{self.father_image_dir}/map_target_map_{self.cur_step}_{t}.jpg',
                map_surroundings_map=f'{self.father_image_dir}/map_surroundings_map_{self.cur_step}_{t}.jpg'
            )
        )
        self.cur_step += 1  

    def step(self, action: int, savefig=True, saveviewfig=True):
        """
        Take one step
        Args:
            action: action, navigation action
        Returns:
            cur_whole_map: current map
            cur_rgb_drone: current drone RGB image
            cur_position: current position
        """
        self.actions.append(action)
        if action == 1: # 前进
            self.cur_pos = Pose4D(
                x=self.cur_pos.x + 5 * math.cos(self.cur_pos.yaw), 
                y=self.cur_pos.y + 5 * math.sin(self.cur_pos.yaw), 
                z=self.cur_pos.z, 
                yaw=self.cur_pos.yaw
            )
        elif action == 3:
            self.cur_pos = Pose4D(
                x=self.cur_pos.x, y=self.cur_pos.y, z=self.cur_pos.z, 
                yaw=self._dump_yaw(self.cur_pos.yaw + math.pi / 6)
            )
        elif action == 2:
            self.cur_pos = Pose4D(
                x=self.cur_pos.x, y=self.cur_pos.y, z=self.cur_pos.z, 
                yaw=self._dump_yaw(self.cur_pos.yaw - math.pi / 6)
            )

        self._generate_photo(savefig=savefig, saveviewfig=saveviewfig)
        return self.cur_whole_map, self.cur_rgb_drone, self.cur_position

    def step_times(self, actions: list):
        """
        Take multiple steps
        Args:
            actions: list of actions
        Returns:
            cur_whole_map: current map
            cur_rgb_drone: current drone RGB image
            cur_position: current position
        """
        n = len(actions)
        for i, action in enumerate(actions):
            self.step(action, savefig=(i == n - 1), saveviewfig=(i == n - 1))
        return self.cur_whole_map, self.cur_rgb_drone, self.cur_position

    def _generate_photo(self, savefig=True, saveviewfig=True):
        """
        Generate photos
        """
        self._get_photo_dirs()
        self.rgb_crop, area = crop_rpg(
            image=self.rgb, map_name=self.episode.map_name, 
            pose=self.cur_pos, shape=self.drone_view_shape, raster=self.raster,
            shape_real_size=self.drone_view_px_size
        )
        
        self.px_trajectory.append(self._get_px(self.cur_pos))
        self.trajectory.append(deepcopy(self.cur_pos))
        tra_re_img = crop_trajectory(
            image=self.map_photo_with_landmark, px_trajectory=self.px_trajectory, area=area,
            savefig=savefig, directions=[math.sin(math.pi/2+self.cur_pos.yaw), math.cos(math.pi/2+self.cur_pos.yaw)]
        )
        if self.cur_step == 2:
            draw_star(self.map_photo_with_landmark, self._get_px(self.start_pose), (0, 0, 0, 255))
        self.map.update_observations(self.cur_pos, self.rgb_crop, None, True)

        self.height_crop = crop_height(
            image=self.height, map_name=self.episode.map_name, 
            pose=self.cur_pos, shape=self.drone_view_shape, raster=self.raster,
            shape_real_size=self.drone_view_px_size
        )

        if savefig:
            plt.imsave(self.photo_dirs[-1].map_photo_path, tra_re_img)
        if saveviewfig:
            plt.imsave(self.photo_dirs[-1].rgb_drone_path, self.rgb_crop)
        # Uncomment as needed for required images
        # Depth view data
        # plt.imsave(self.photo_dirs[-1].height_drone_path, height_crop)
        # Map data
        # map_data = self.map.to_array()
        # plt.imsave(self.photo_dirs[-1].map_current_view_area, map_data[0])
        # plt.imsave(self.photo_dirs[-1].map_explored_area, map_data[1])
        # plt.imsave(self.photo_dirs[-1].map_landmark_map, map_data[2])
        # plt.imsave(self.photo_dirs[-1].map_target_map, map_data[3])
        # plt.imsave(self.photo_dirs[-1].map_surroundings_map, map_data[4])

    def _get_px(self, pose):
        """
        Get pixel coordinates
        Args:
            pose: position
        Returns:
            [center_y, center_x]: center coordinates
        """
        center_x, center_y = self.raster.index(pose.x, pose.y)
        return [int(center_y), int(center_x)]

    def clean_fater_image_dir(self):
        for file in os.listdir(self.father_image_dir):
            os.remove(os.path.join(self.father_image_dir, file))
        os.rmdir(self.father_image_dir)

    @staticmethod
    def _dump_yaw(yaw):
        return (yaw + math.pi) % (2 * math.pi) - math.pi

    @property
    def map(self):
        """
        Get map
        Returns:
            data.map: map
        """
        return self.data.map
    
    @property
    def rgb(self):
        """
        Get RGB image
        Returns:
            data.rgb: RGB image
        """
        return self.data.rgb
    
    @property
    def height(self):
        """
        Get height (depth) image
        Returns:
            data.height: height image
        """
        return self.data.height
    
    @property
    def episode(self):
        """
        Get episode data
        Returns:
            data.episode: episode
        """
        return self.data.episode

    @property
    def px_list(self):
        """
        Get list of pixel coordinates
        Returns:
            data.px_list: pixel list
        """
        return self.data.px_list
    
    @property
    def start_pose(self):
        """
        Get starting position
        Returns:
            data.start_pose: start position
        """
        return self.episode.start_pose
    
    @property
    def target_description(self):
        """
        Get target description
        Returns:
            data.target_description: description
        """
        return self.episode.target_description

    @property
    def raster(self):
        """
        Get raster object
        Returns:
            data.raster: raster
        """
        return self.data.raster
    
    @property
    def cur_whole_map(self):
        """
        Get the current map image
        Returns:
            cur_whole_map: current map
        """
        return self.photo_dirs[-1].map_photo_path

    @property
    def cur_rgb_drone(self):
        """
        Get the current drone RGB image
        Returns:
            cur_rgb_drone: current drone RGB image
        """
        return self.photo_dirs[-1].rgb_drone_path

    @property
    def cur_height_drone(self):
        """
        Get the current drone height (depth) image
        Returns:
            cur_height_drone: current drone height image
        """
        return self.photo_dirs[-1].height_drone_path

    @property
    def cur_map_current_view_area(self):
        """
        Get the map of the current view area
        Returns:
            cur_map_current_view_area: current view area map
        """
        return self.photo_dirs[-1].map_current_view_area

    @property
    def cur_map_explored_area(self):
        """
        Get the explored area map
        Returns:
            cur_map_explored_area: explored area map
        """
        return self.photo_dirs[-1].map_explored_area

    @property
    def cur_map_landmark_map(self):
        """
        Get the landmark map
        Returns:
            cur_map_landmark_map: landmark map
        """
        return self.photo_dirs[-1].map_landmark_map

    @property
    def cur_map_target_map(self):
        """
        Get the target map
        Returns:
            cur_map_target_map: target map
        """
        return self.photo_dirs[-1].map_target_map

    @property
    def cur_map_surroundings_map(self):
        """
        Get the surroundings map
        Returns:
            cur_map_surroundings_map: surroundings map
        """
        return self.photo_dirs[-1].map_surroundings_map

    @property
    def cur_position(self):
        """
        Get the current position
        Returns:
            [self.cur_pos.x, self.cur_pos.y, self.cur_pos.z, self.cur_pos.yaw]: current position [x, y, z, yaw (in degrees)]
        """
        return [self.cur_pos.x, self.cur_pos.y, self.cur_pos.z, self.cur_pos.yaw * 180 / math.pi]

    @property
    def px_real_size(self):
        """
        Get the actual size of each pixel
        Returns:
            [abs(self.raster.transform.a), abs(self.raster.transform.e)]: pixel real-world size in meters
        """
        return [abs(self.raster.transform.a), abs(self.raster.transform.e)]

    @property
    def real_vision_size(self):
        """
        Get the real-world size of the drone's field of view
        Returns:
            [width_in_meters, height_in_meters]: real-world view size
        """
        return [abs(self.raster.transform.a) * self.drone_view_shape[0], abs(self.raster.transform.e) * self.drone_view_shape[1]]

    @property
    def target_px(self):
        """
        Get the pixel coordinates of the target
        Returns:
            target_px: pixel coordinates of the target
        """
        return self._get_px(self.episode.target_position)

    @property
    def top_left(self):
        """
        Get the top-left corner coordinates of the map in real-world units
        Returns:
            [x, y]: top-left corner
        """
        return [self.raster.transform.c, self.raster.transform.f]

    @property
    def murk_actions(self):
        """
        Get murk (teacher) actions
        Returns:
            murk_actions: list of teacher actions
        """
        return self.episode.teacher_actions

    @property
    def map_name(self):
        """
        Get the name of the map
        Returns:
            map_name: map name
        """
        return self.episode.map_name

    