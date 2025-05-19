import os
import re
import ast
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json

from open_r1.vlm_modules import *

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging

from openai import OpenAI
import ast
import numpy as np
import math

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )

def euclidean_distance(cur_pose, target_location):
    cur_x, cur_y = cur_pose
    target_x, target_y = target_location
    
    distance = math.sqrt((target_x - cur_x) ** 2 + (target_y - cur_y) ** 2)
    return distance

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        try:
            sol = ast.literal_eval(sol)
        except Exception as e:
            print(e)

        
        target_position = sol['target_position']
        matches = re.findall(r'"target_location"\s*:\s*\[(\d+), (\d+)\]', content)
        if matches:
            pred_position = list(map(int, matches[0]))
        else:
            pred_position = [0, 0]
        px_distance = euclidean_distance(pred_position, target_position)
        
        if px_distance <= 800:
            soft_reward = np.exp(-(max(px_distance, 200) - 200) / 100.0)
        else:
            soft_reward = 0
        
        
        reward = soft_reward
        
        
        
        # iou reward
        landmark_bboxes = sol['landmark_bbox']
        matches = re.findall(r'"landmark_bbox"\s*:\s*\[(\d+), (\d+), (\d+), (\d+)\]', content)

        if matches:
            iou_reward = 0
            pred_landmark_bbox = list(map(int, matches[0]))
            for landmark_bbox in landmark_bboxes:
                landmark_iou = iou(landmark_bbox, pred_landmark_bbox)
                iou_reward = max(iou_reward, landmark_iou)
            reward += iou_reward
        else:
            iou_reward = 0

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 3:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"soft_reward: {soft_reward},    iou_reward: {iou_reward}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")    
                    f.write(f"\n\n\n\n\n\n")    
        
        rewards.append(reward) 

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        reward = 0
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if think_match and answer_match:
            reward += 0.5

            
        if think_match:
            think_content = think_match.group(1).strip()
            landmark_matches = re.findall(r'"landmark_bbox"\s*:\s*\[(\d+), (\d+), (\d+), (\d+)\]', think_content)
            if landmark_matches:
                reward += 0.25

        if answer_match:
            answer_content = answer_match.group(1).strip()
            target_matches = re.findall(r'"target_location"\s*:\s*\[(\d+), (\d+)\]', answer_content)
            if target_matches:
                reward += 0.25

    rewards.append(reward)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

from torch.utils.data import Dataset
from PIL import Image
import random

SYSTEM_PROMPT = "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization."
class CitynavDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(CitynavDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        with open(data_path, 'r') as f:
            citynav_data = json.load(f)
        for step_data in citynav_data:
            item = {}
            image_path = script_args.image_folders + step_data['image_path']
            item['image_path'] = image_path
            item['start_position'] = step_data['start_position']
            item['problem'] = step_data['target_description'] 
            item['solution'] = f"{{'landmark_bbox': {step_data['landmark_bbox']}, 'target_position': {step_data['target_position']}}}" 
            self.list_data_dict.append(item)

    def __len__(self):
        return len(self.list_data_dict)
    
    def get_prompt(self, instruction, cur_pose):
        prompt = f"""
    [Mission Objective]  
    Your mission is to locate a specific target described via natural language instructions.

    [Details of the Target]  
    {instruction}

    [Environmental Perception]  
    - The UAV's current position is indicated by the starting point of an arrow in the image, with its orientation represented by the arrow's direction.  
    - The yellow box outlines the UAV's current field of view, centered at pixel coordinates: cur_pose = {cur_pose}.  
    - Street-related landmark regions are visually marked using red masks.

    [Operational Guidance]  
    - The target is always positioned near a red-masked street landmark.  
    - Use both the instruction and the visual scene to identify the most relevant red-masked landmark region.  
    - Reason about the likely relative position of the target with respect to that landmark.

    [Output Format Specification]  
    - Present your reasoning within `<think>` and `</think>` tags.  
    For example, your reasoning may include the following elements:  
    - A semantic interpretation of the instruction.  
    - Identification of the correct landmark region.  
    - The bounding box of that region in the format:  
        `{{"landmark_bbox": [x1, y1, x2, y2]}}`  

    - Then provide your final answer within `<answer>` and `</answer>` tags as:  
    `{{"target_location": [x, y]}}`
    """
    
        return prompt

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }

        def make_conversation_image(example):
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text":  self.get_prompt(example['problem'], example['start_position'])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_path = example['image_path']
        image = Image.open(image_path).convert("RGB")
        

        return {
            'image': image,
            'image_path': image_path,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'] if 'image_path' in example else make_conversation(example)['prompt'],
        }




def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type="default")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    dataset = CitynavDataset(script_args.dataset_name, script_args)
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
