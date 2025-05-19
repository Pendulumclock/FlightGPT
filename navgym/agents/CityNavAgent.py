import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from pydantic import BaseModel
from navgym.models.NavGym import action_dict, action_list
from openai import OpenAI


def get_prompt(instruction, cur_pose):
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


class GPTInfo(BaseModel):
    api_key: str
    api_version: str
    api_base: str
    model: str


class GPTAgent:
    def __init__(
            self, api_key, api_version, api_base, model, 
            system_prompt, target_description, drone_see_shape,
            scale, top_left
        ):
        self.gpt_info = GPTInfo(
            api_key = api_key,
            api_base = api_base,
            api_version =api_version,
            model = model
        )
        self.system_prompt = system_prompt
        self.target_description = target_description
        self.drone_see_shape = drone_see_shape
        self.scale = scale
        self.top_left = top_left


    def act(self, cur_whole_map, cur_rgb_drone, cur_position):
        result = self._gpt4o_imagefile(
            map_file=cur_whole_map, 
            view_file=cur_rgb_drone, 
            system_prompt=self.system_prompt, 
            prompt=get_prompt(
                instruction=self.target_description, cur_pose=cur_position
            )
        )
        response = result.choices[0].message.content
        return response
    


    @staticmethod
    def _local_image_to_data_url(image_path):
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def _gpt4o_imagefile(self, map_file, view_file, system_prompt, prompt):
        """
        Gpt-4o model
        """
        client = OpenAI(
            base_url=self.gpt_info.api_base,
            api_key=self.gpt_info.api_key
        )

        response = client.chat.completions.create(
            model=self.gpt_info.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._local_image_to_data_url(map_file)},
                        }
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.0,
        )

        return response