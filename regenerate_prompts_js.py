#!/usr/bin/env python3
"""
Regenerate Genie/prompts.js from prompts.py
"""
import json
import sys
import os
from pathlib import Path

# Add current directory to path to import prompts.py
sys.path.insert(0, os.path.dirname(__file__))

from prompts import (
    solo_prompts_dict, couple_man_unseen_prompts_dict, couple_man_seen_prompts_dict, 
    multiple_women_prompts_dict, multiple_man_unseen_prompts_dict, multiple_man_seen_prompts_dict, 
    multistep_prompts_dict, vid_solo_prompts_dict, vid_couple_prompts_dict, 
    vid_multiple_prompts_dict, vid_multistep_prompts_dict, vid_environment_prompts_dict, 
    vid_custom_prompts_dict, vid_multiple_man_unseen_prompts_dict, vid_multiple_man_seen_prompts_dict
)

# Create the combined prompts object
genie_prompts = {
    "solo_prompts_dict": solo_prompts_dict,
    "couple_man_unseen_prompts_dict": couple_man_unseen_prompts_dict,
    "couple_man_seen_prompts_dict": couple_man_seen_prompts_dict,
    "multiple_women_prompts_dict": multiple_women_prompts_dict,
    "multiple_man_unseen_prompts_dict": multiple_man_unseen_prompts_dict,
    "multiple_man_seen_prompts_dict": multiple_man_seen_prompts_dict,
    "multistep_prompts_dict": multistep_prompts_dict,
    "vid_solo_prompts_dict": vid_solo_prompts_dict,
    "vid_couple_prompts_dict": vid_couple_prompts_dict,
    "vid_multiple_prompts_dict": vid_multiple_prompts_dict,
    "vid_multistep_prompts_dict": vid_multistep_prompts_dict,
    "vid_environment_prompts_dict": vid_environment_prompts_dict,
    "vid_custom_prompts_dict": vid_custom_prompts_dict,
    "vid_multiple_man_unseen_prompts_dict": vid_multiple_man_unseen_prompts_dict,
    "vid_multiple_man_seen_prompts_dict": vid_multiple_man_seen_prompts_dict
}

# Generate the JavaScript file
js_content = f"""// Auto-generated from prompts.py – keep in sync with the server's presets.
window.GENIE_PROMPTS = {json.dumps(genie_prompts, indent=2)};
"""

# Write to the Genie folder
genie_folder = Path(__file__).parent / "Genie"
output_file = genie_folder / "prompts.js"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(js_content)

print(f"Successfully regenerated {output_file}")