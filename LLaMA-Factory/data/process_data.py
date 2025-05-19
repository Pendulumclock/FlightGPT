import json


def get_prompt(instruction, cur_pose):
        
    prompt = f'''[Mission Objective]  
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
    
    '''
    return prompt

with open('./LLaMA-Factory/data/cleaned_data.json', 'r') as f:
    data = json.load(f)
processed_data = []
for index in range(len(data)):
    # data[index]['messages'][1]["content"] =data[index]['messages'][1]["content"].split('### Final Answer')[-1].strip()
    message = dict()
    system_content = dict()
    system_content['content'] = 'You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization.'
    system_content['role'] = 'system'
    user_content = dict()
    user_content['content'] =get_prompt(data[index]['target_description'], data[index]['start_position'])
    user_content['content'] = user_content['content']+'\n<image>'
    user_content['role'] = 'user'
    message['messages'] = []
    message['messages'].append(system_content)
    message['messages'].append(user_content)
    message['images'] = []
    message['images'].append(data[index]['image_path'])
    assistant_content = dict()
    assistant_content['content'] = data[index]['content_text']
    assistant_content['role'] = 'assistant'
    message['messages'].append(assistant_content)
    processed_data.append(message)
    
print(len(processed_data))
with open('./LLaMA-Factory/data/cleaned_final.json', 'w') as f:
    json.dump(processed_data, f, indent=4)
