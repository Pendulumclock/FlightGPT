import os
import sys
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.GPTAgent_4o import GPTAgent
from navgym.tools.EvalTools import eval_goal_predictor, eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, view_area_corners
from concurrent.futures import ThreadPoolExecutor, as_completed


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
cropclient.load_image_cache()


#Your api key
API_CONFIG_LIST = [
    {
        "api_key": 'Your_key_1',
        "api_base": 'Your_base_1',
        "api_version": "2024-03-01-preview",
        "model": "gpt-4o",
        "system_prompt": "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization."
    },
    {
        "api_key": 'Your_key_2',
        "api_base": 'Your_base_2',
        "api_version": "2024-03-01-preview",
        "model": "gpt-4o",
        "system_prompt": "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization."
    },
]
SAVE_PATH = "./experiment"


def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def parse_bbox(result_str, key="landmark_bbox"):
    pattern = fr'"{key}"\s*:\s*\[(\d+), (\d+), (\d+), (\d+)\]'
    match = re.search(pattern, result_str)
    return list(map(int, match.groups())) if match else [0, 0, 0, 0]

def parse_location(result_str):
    match = re.search(r'"target_location"\s*:\s*\[(\d+), (\d+)\]', result_str)
    return list(map(int, match.groups())) if match else [0, 0]

def visualize_prediction(navGym, source_path, landmark_box, target_pred, true_target, save_path):
    image = cv2.imread(source_path)

    # Draw all landmarks
    for landmark in navGym.map.landmark_map.landmarks:
        top_left = navGym._get_px(landmark.bbox_corners[0])
        bottom_right = navGym._get_px(landmark.bbox_corners[2])
        cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 255), thickness=2)

    cv2.rectangle(image, (landmark_box[0], landmark_box[1]), (landmark_box[2], landmark_box[3]), (0, 0, 255), 2)
    cv2.circle(image, tuple(target_pred), radius=30, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, tuple(true_target), radius=30, color=(255, 0, 0), thickness=-1)

    create_dir(save_path)
    cv2.imwrite(save_path, image)

def compute_pose(navGym, predicted_px, true_start_px, map_name):
    if predicted_px == [0, 0]:
        return navGym.start_pose

    dx, dy = predicted_px[0] - true_start_px[0], predicted_px[1] - true_start_px[1]
    world_x = dx / 10 + navGym.episode.start_pose.x
    world_y = navGym.episode.start_pose.y - dy / 10
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    corners = view_area_corners(base_pose, GROUND_LEVEL[map_name])
    depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
    center_depth = depth_img[45:55, 45:55].mean()
    refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
    return refined_pose



from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
def move(pose: Pose4D, dst: Pose4D, iterations: int):

    dst = Point3D(dst.x, dst.y, pose.z)
    trajectory = []
    for _ in range(iterations):
        action = lookahead_discrete_action(pose, [dst])
        if action.name == 'STOP':
            return trajectory
        pose = _moved_pose(pose, *action.value)
        trajectory.append(pose)
    return trajectory

def calculate_mean_metrics(results, nums):
    total_nums = nums['easy'] + nums['medium'] + nums['hard']
    NE = results['easy'].mean_final_pos_to_goal_dist * nums['easy']/total_nums + \
        results['medium'].mean_final_pos_to_goal_dist * nums['medium']/total_nums + \
        results['hard'].mean_final_pos_to_goal_dist * nums['hard']/total_nums

    SR = results['easy'].success_rate_final_pos_to_goal * nums['easy']/total_nums + \
        results['medium'].success_rate_final_pos_to_goal * nums['medium']/total_nums + \
        results['hard'].success_rate_final_pos_to_goal * nums['hard']/total_nums
        
    OSR = results['easy'].success_rate_oracle_pos_to_goal  * nums['easy']/total_nums + \
        results['medium'].success_rate_oracle_pos_to_goal  * nums['medium']/total_nums + \
        results['hard'].success_rate_oracle_pos_to_goal  * nums['hard']/total_nums

    SPL = results['easy'].success_rate_weighted_by_path_length  * nums['easy']/total_nums + \
        results['medium'].success_rate_weighted_by_path_length  * nums['medium']/total_nums + \
        results['hard'].success_rate_weighted_by_path_length  * nums['hard']/total_nums
    
    return NE, SR, OSR, SPL

def process_sample(idx, citynavData, api_config, step, action_num):
    try:
        pose_history = []
        cur_trajectory = []
        cur_citynavData = citynavData[idx]
        for _ in range(step):
            if pose_history != []:
                cur_citynavData.episode.teacher_trajectory[0] = pose_history[-1]
            navGym = NavGym(cur_citynavData)
            start_pose = navGym.start_pose
            
            agent = GPTAgent(
                api_key=api_config["api_key"],
                api_base=api_config["api_base"],
                api_version=api_config["api_version"],
                model=api_config["model"],
                system_prompt=api_config["system_prompt"],
                target_description=navGym.target_description,
                drone_see_shape=navGym.drone_view_shape,
                scale=navGym.px_real_size,
                top_left=navGym.top_left
            )

            map_name = navGym.episode.id[0]
            result_str = agent.act(
                cur_whole_map=navGym.cur_whole_map,
                cur_rgb_drone=navGym.cur_rgb_drone,
                cur_position=navGym._get_px(start_pose),
                history_actions=navGym.actions
            )

            landmark_bbox = parse_bbox(result_str, "landmark_bbox")
            target_pred_px = parse_location(result_str)
            true_start_px = navGym.px_trajectory[0]
            true_target_px = navGym.target_px
            

            save_path = f"{SAVE_PATH}/image_twostep/{os.path.basename(navGym.cur_whole_map)}"
            # visualize_prediction(navGym, navGym.cur_whole_map, landmark_bbox, target_pred_px, true_target_px, save_path)

            pred_pose = compute_pose(navGym, target_pred_px, true_start_px, map_name)
            
            if pose_history == []:
                cur_trajectory = [start_pose]
                move_trajectory = move(start_pose, pred_pose, action_num)
                if len(move_trajectory) > 0:
                    pose_history.append(move_trajectory[-1])
                cur_trajectory.extend(move_trajectory)
            else:
                move_trajectory = move(start_pose, pred_pose, action_num)
                
                if len(move_trajectory) > 0:
                    pose_history.append(move_trajectory[-1])
                cur_trajectory.extend(move_trajectory)

        return (citynavData.episodes[idx].id, cur_trajectory, None, navGym.father_image_dir)

    except Exception as e:
        return (None, None, idx, None)


def run_nav_gym(citynavData, split):
    trajectory = {}
    errors = []
    image_dir = None
    step_num = 2    #total steps that agent take
    action_num = 75     #actions per step

    with ThreadPoolExecutor(max_workers=len(API_CONFIG_LIST)) as executor:
        futures = []
        for i in range(len(citynavData)):
            config = API_CONFIG_LIST[i % len(API_CONFIG_LIST)]
            futures.append(executor.submit(process_sample, i, citynavData, config, step_num, action_num))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {split}"):
            episode_id, cur_trajectory, error_idx, img_dir = future.result()
            if episode_id:
                trajectory[episode_id] = cur_trajectory
                image_dir = img_dir
            elif error_idx is not None:
                errors.append(error_idx)

    return trajectory, errors, image_dir

# -------- 主流程 -------- #
def main():
    results = {}
    nums = {}
    for split in ["easy", "medium", "hard"]:
        data_path = f"./data/citynav/citynav_val_unseen_{split}.json"
        citynavData = CityNavData(data_path)

        traj, errors, image_dir = run_nav_gym(citynavData, split)
        print(f"Image Dir: {image_dir}, Errors: {errors}")

        with open(f"{SAVE_PATH}/data/citynav_val_unseen_{split}.pkl", "wb") as f:
            pickle.dump(traj, f)

        episodes = [ep for ep in citynavData.episodes if ep.id in traj]
        metrics = eval_planning_metrics(episodes, traj)
        print(f"{split} result:", metrics)
        results[split] = metrics
        nums[split] = len(episodes)
    
    NE, SR, OSR, SPL = calculate_mean_metrics(results, nums)
    print("Final Results:", results)
    print(f'NE:{NE}\nSR:{SR}\nOSR:{OSR}\nSPL:{SPL}')

if __name__ == "__main__":
    main()
