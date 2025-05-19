from tqdm import tqdm

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.observation import cropclient
from gsamllavanav.dataset.episode import EpisodeID
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Point2D, Point3D, Pose4D, bbox_corners_to_position, crwh_to_global_bbox, view_area_corners
from gsamllavanav.maps.gsam_map import GSamMap, GSamParams
from gsamllavanav import vlmodel
from gsamllavanav import som
import matplotlib.pyplot as plt

def goal_selection_gdino(
    args: ExperimentArgs,
    pred_goal_logs: dict[EpisodeID, list[Point2D]]
) -> dict[EpisodeID, Pose4D]:
    
    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()
    gsam_params = GSamParams(
        args.gsam_use_segmentation_mask, args.gsam_use_bbox_confidence,
        args.gsam_box_threshold, args.gsam_text_threshold,
        args.gsam_max_box_size, args.gsam_max_box_area,
    )

    predicted_positions = {}  
    for (map_name, obj_id, desc_id), pred_positions in tqdm(pred_goal_logs.items(), desc='selecting target bbox', unit='trajectory'):
        
        final_pred_pose = Pose4D(*pred_positions[-1], args.altitude + GROUND_LEVEL[map_name], 0)
        # final_pred_pose = pred_positions[-1]
        rgb = cropclient.crop_image(map_name, final_pred_pose, (int(args.altitude*10), int(args.altitude*10)), 'rgb')

        target_object = objects[map_name][obj_id]
        target_name = target_object.processed_descriptions[desc_id].target
        
        gsam_map = GSamMap(map_name, (240, 240), 240/410, [target_name], gsam_params)
        gsam_map.update_observation(final_pred_pose, rgb)
        pred_pos = bbox_corners_to_position(gsam_map.max_confidence_bbox, gsam_map.ground_level)

        camera_z = GROUND_LEVEL[map_name] + args.altitude
        camera_pose = Pose4D(pred_pos.x, pred_pos.y, camera_z, 0)
        depth = cropclient.crop_image(map_name, camera_pose, (100, 100), 'depth')
        z_around_center = camera_pose.z - depth[45:55, 45:55].mean()
        final_pose = Pose4D(pred_pos.x, pred_pos.y, z_around_center + 5, 0)

        predicted_positions[(map_name, obj_id, desc_id)] = final_pose
    
    return predicted_positions


def goal_selection_llava(
    args: ExperimentArgs,
    pred_goal_logs: dict[EpisodeID, list[Point2D]]
) -> dict[EpisodeID, Pose4D]:
    
    NOT_IN_IMAGE = -1
    INVALID_RESPONSE = -2

    vlmodel.load_model('llava-v1.6-34b')
    som.load_model('semantic-sam')
    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()

    predicted_positions = {}
    label_num = [0, 0, 0]
    for (map_name, obj_id, desc_id), pred_goals in tqdm(pred_goal_logs.items(), desc='selecting target bbox', unit='trajectory'):
        for i in range(len(pred_goals)):
            pred_goals[i] = Pose4D(*pred_goals[i], args.altitude + GROUND_LEVEL[map_name], 0)
        # (x, y), yaw = pred_goals[-1], 0.
        ground_level = GROUND_LEVEL[map_name]
        target_object = objects[map_name][obj_id]
        # camera_pose = Pose4D(x, y, args.altitude + ground_level, yaw)
        camera_pose = pred_goals[-1]
        
        rgb = cropclient.crop_image(map_name, camera_pose, (args.altitude*10, args.altitude*10), 'rgb')
        output_path = f'/mnt/vepfs/fs_users/Djinhan/LLN/grounding/output_test/rgb/{map_name}_{obj_id}_{desc_id}.jpg'
        plt.imsave(output_path, rgb)
        annotated_rgb, masks = som.annotate(rgb, 'semantic-sam', [4])


        # TODO
        # goal_sel_img = '/mnt/vepfs/fs_users/caihengxing/tanjj/LLN/R1PhotoData/llava_images/goal_sel.jpg'
        # plt.imsave(goal_sel_img, rgb)
        # goal_sel_img_annotated = '/mnt/vepfs/fs_users/caihengxing/tanjj/LLN/R1PhotoData/llava_images/goal_sel_annotated.jpg'
        # plt.imsave(goal_sel_img_annotated, annotated_rgb)
        # print(target_object.descriptions)

        output_path = f'/mnt/vepfs/fs_users/Djinhan/LLN/grounding/output_test/annotated_rgb/{map_name}_{obj_id}_{desc_id}.jpg'
        plt.imsave(output_path, annotated_rgb)

        
        #prompt = f"Answer the label number of the object that the following text describes. If the ojbect is not present in the image, answer {NOT_IN_IMAGE} instead.\n"
        prompt = f"Carefully analyze the given description and determine whether an object in the image matches the described category. First, verify if any object belongs to the specified category. If no matching object is found, answer {NOT_IN_IMAGE}. Otherwise, provide the label number of the best-matching object. Output only one number.\n"
        prompt += f":\n{target_object.descriptions[desc_id]}"
        #prompt += f":\n{target_object.processed_descriptions[desc_id].target}"
        response = vlmodel.query(annotated_rgb, prompt)

        try:
            label = int(response)
        except ValueError:
            label = INVALID_RESPONSE
        # print(label)
        if label == - 2:
            label_num[0] += 1
        elif label == -1:
            label_num[1] += 1
        elif label > len(masks):
            label_num[2] += 1
        bbox_corners = crwh_to_global_bbox(masks[label - 1]['bbox'], rgb.shape[:2], camera_pose, ground_level) if 0 < label <= len(masks) else view_area_corners(camera_pose, ground_level)
        pred_pos = bbox_corners_to_position(bbox_corners, ground_level) if 0 < label <= len(masks) else camera_pose.xyz

        camera_z = GROUND_LEVEL[map_name] + args.altitude
        camera_pose = Pose4D(pred_pos.x, pred_pos.y, camera_z, 0)
        depth = cropclient.crop_image(map_name, camera_pose, (100, 100), 'depth')
        z_around_center = camera_pose.z - depth[45:55, 45:55].mean()
        final_pose = Pose4D(pred_pos.x, pred_pos.y, z_around_center + 5, 0)

        predicted_positions[(map_name, obj_id, desc_id)] = final_pose
    print(label_num)
    return predicted_positions