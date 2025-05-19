import random
import re
import os

# Helper function to read lines from a file
def read_lines_from_file(filepath):
    """Reads lines from a file, stripping whitespace. Handles file not found."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return []

# Helper function to parse replacement rules
def parse_rules(rules_str):
    """Parses 'old->new' rules from a multiline string."""
    rules = []
    for line in rules_str.splitlines():
        if '->' in line:
            old, new = line.split('->', 1)
            rules.append((old.strip(), new.strip()))
    return rules

# Helper function to apply rules to a single prompt string
def apply_rules_to_prompt(prompt_text, rules):
    """Applies replacement rules to a prompt string."""
    for old, new in rules:
        # Use \b for word boundaries to avoid partial replacements
        # re.escape handles special characters in 'old'
        pattern = r'\b' + re.escape(old) + r'\b'
        prompt_text = re.sub(pattern, new, prompt_text)
    return prompt_text

# Helper function to clean up commas and extra spaces
def cleanup_prompt_tags(prompt_text):
    """Cleans up comma formatting in a tag string."""
    # Replace multiple commas (with optional spaces around them) with a single comma
    prompt_text = re.sub(r'\s*,\s*(?:,\s*)+', ',', prompt_text)
    # Remove leading/trailing commas and spaces
    prompt_text = prompt_text.strip(' ,')
    # Remove spaces around single commas
    prompt_text = re.sub(r'\s*,\s*', ',', prompt_text)
    # Further cleanup for potential empty tags if rules removed content
    # e.g. "tag1,,tag2" or "tag1, ,tag2" should become "tag1,tag2"
    tags = [tag.strip() for tag in prompt_text.split(',') if tag.strip()]
    return ', '.join(tags) # Rejoin with consistent ", "

# Core logic function
def generate_prompts_logic(s1_count, s2_count, s3_count,
                           s1_outfit, s2_outfit, s3_outfit,
                           rules_str, seed, base_path):
    """
    Generates prompts based on sections, outfits, rules, and seed.
    base_path is the directory where 'sections' folder is located.
    """
    if seed is None or seed < 0: # ComfyUI often uses -1 for random seed
        rng = random.Random() # New instance for non-deterministic
    else:
        rng = random.Random(seed)

    section_files = {
        1: os.path.join(base_path, "sections", "section1.txt"),
        2: os.path.join(base_path, "sections", "section2.txt"),
        3: os.path.join(base_path, "sections", "section3.txt")
    }

    s1_prompts_all = read_lines_from_file(section_files[1])
    s2_prompts_all = read_lines_from_file(section_files[2])
    s3_prompts_all = read_lines_from_file(section_files[3])

    # Determine actual number of prompts to sample (min of requested and available)
    num_s1_to_sample = min(s1_count, len(s1_prompts_all))
    num_s2_to_sample = min(s2_count, len(s2_prompts_all))
    num_s3_to_sample = min(s3_count, len(s3_prompts_all))

    if s1_count > len(s1_prompts_all):
        print(f"Warning: Requested {s1_count} prompts for section 1, but only {len(s1_prompts_all)} available. Using {len(s1_prompts_all)}.")
    if s2_count > len(s2_prompts_all):
        print(f"Warning: Requested {s2_count} prompts for section 2, but only {len(s2_prompts_all)} available. Using {len(s2_prompts_all)}.")
    if s3_count > len(s3_prompts_all):
        print(f"Warning: Requested {s3_count} prompts for section 3, but only {len(s3_prompts_all)} available. Using {len(s3_prompts_all)}.")

    chosen_s1 = rng.sample(s1_prompts_all, num_s1_to_sample) if num_s1_to_sample > 0 else []
    chosen_s2 = rng.sample(s2_prompts_all, num_s2_to_sample) if num_s2_to_sample > 0 else []
    chosen_s3 = rng.sample(s3_prompts_all, num_s3_to_sample) if num_s3_to_sample > 0 else []

    parsed_rules = parse_rules(rules_str)
    final_prompts_list = []

    # Process Section 1
    for p in chosen_s1:
        combined = f"{s1_outfit.strip()}, {p}" if s1_outfit.strip() else p
        processed = apply_rules_to_prompt(combined, parsed_rules)
        cleaned = cleanup_prompt_tags(processed)
        if cleaned: # Only add if not empty after cleanup
            final_prompts_list.append(cleaned)

    # Process Section 2
    for p in chosen_s2:
        combined = f"{s2_outfit.strip()}, {p}" if s2_outfit.strip() else p
        processed = apply_rules_to_prompt(combined, parsed_rules)
        cleaned = cleanup_prompt_tags(processed)
        if cleaned:
            final_prompts_list.append(cleaned)

    # Process Section 3
    for p in chosen_s3:
        combined = f"{s3_outfit.strip()}, {p}" if s3_outfit.strip() else p
        processed = apply_rules_to_prompt(combined, parsed_rules)
        cleaned = cleanup_prompt_tags(processed)
        if cleaned:
            final_prompts_list.append(cleaned)
            
    return " / ".join(final_prompts_list)


class PromptGeneratorNode:
    def __init__(self):
        # Get the directory of the current script
        # This is important for finding the 'sections' folder relative to the node
        self.base_dir = os.path.dirname(os.path.realpath(__file__))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s1_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s2_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s3_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s1_outfit": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "s2_outfit": ("STRING", {"multiline": True, "default": ""}),
                "s3_outfit": ("STRING", {"multiline": True, "default": ""}),
                "replacement_rules": ("STRING", {
                    "multiline": True,
                    "default": "1boy->1man, pale_skin\nsweat->sweat, water_drops, wet\nnavel->"
                }),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}), # -1 for random in Comfy often
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_prompts",)
    FUNCTION = "generate_prompts"
    CATEGORY = "Prompt Utilities/PackCREATOR_BOLADEX"

    def generate_prompts(self, s1_prompt_count, s2_prompt_count, s3_prompt_count,
                         s1_outfit, s2_outfit, s3_outfit,
                         replacement_rules, seed):
        
        generated_string = generate_prompts_logic(
            s1_prompt_count, s2_prompt_count, s3_prompt_count,
            s1_outfit, s2_outfit, s3_outfit,
            replacement_rules, seed,
            self.base_dir # Pass the base directory of the node
        )
        return (generated_string,)

# Example usage (for testing outside ComfyUI)
if __name__ == "__main__":
    # Create dummy section files in a 'sections' subdirectory relative to this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sections_dir = os.path.join(script_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)

    with open(os.path.join(sections_dir, "section1.txt"), "w") as f:
        f.write("blue_sky, 1girl, solo, sweat\n")
        f.write("sunny_day, landscape, mountains, navel\n")
        f.write("forest_path, green_trees, sunlight, 1boy\n")

    with open(os.path.join(sections_dir, "section2.txt"), "w") as f:
        f.write("detailed_face, happy_smile, looking_at_viewer, sweat\n")
        f.write("dynamic_pose, action_shot, blurred_background\n")

    with open(os.path.join(sections_dir, "section3.txt"), "w") as f:
        f.write("cyberpunk_city, neon_lights, rain, 1boy\n")

    print(f"Testing with base_dir: {script_dir}")

    # Test the core logic function
    result = generate_prompts_logic(
        s1_count=2, s2_count=1, s3_count=1,
        s1_outfit="s1_style, red_dress",
        s2_outfit="s2_style",
        s3_outfit="", # Empty outfit
        rules_str="1boy->1man, pale_skin\nsweat->sweat, water_drops, wet\nnavel->\nred_dress->blue_dress",
        seed=42,
        base_path=script_dir # Use the script's directory for testing
    )
    print("\n--- Generated Prompts (Logic Test) ---")
    print(result)
    print("-------------------------------------\n")

    # Test the node class (simulating ComfyUI call)
    node_instance = PromptGeneratorNode()
    output_tuple = node_instance.generate_prompts(
        s1_prompt_count=1,
        s2_prompt_count=1,
        s3_prompt_count=0,
        s1_outfit="ultra_detailed, 8k",
        s2_outfit="cinematic",
        s3_outfit="sketch",
        replacement_rules="1girl->1woman\nmountains->snowy_mountains",
        seed=123
    )
    print("--- Generated Prompts (Node Test) ---")
    print(output_tuple[0])
    print("-----------------------------------\n")

    # Test with more prompts than available
    result_overflow = generate_prompts_logic(
        s1_count=10, s2_count=1, s3_count=0,
        s1_outfit="overflow_style", s2_outfit="", s3_outfit="",
        rules_str="", seed=1, base_path=script_dir
    )
    print("--- Generated Prompts (Overflow Test) ---")
    print(result_overflow)
    print("---------------------------------------\n")