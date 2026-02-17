import json
import os

split_path = '/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/Data/oxford_pets/split_zhou_OxfordPets.json'

try:
    with open(split_path, 'r') as f:
        data = json.load(f)
        
    print(f"Keys: {data.keys()}")
    for key in data:
        print(f"{key} count: {len(data[key])}")
        
except Exception as e:
    print(f"Error reading json: {e}")
