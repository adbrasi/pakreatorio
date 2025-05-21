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

    # Process Section 2 (MODIFIED LOGIC HERE)
    if chosen_s2: # Only process if there are s2 prompts
        num_s2_total = len(chosen_s2)
        # Calculate the number of prompts to get s2_outfit (first 40%)
        # Ensure at least one prompt gets s2_outfit if num_s2_total is small but > 0, and s2_outfit is intended
        # However, strict 40% means integer division will handle small numbers appropriately.
        # For example, if num_s2_total is 1 or 2, 0.4 * num_s2_total will be 0.
        # If num_s2_total is 3, 0.4 * 3 = 1.2 -> 1 prompt gets s2_outfit.
        # If num_s2_total is 5, 0.4 * 5 = 2.0 -> 2 prompts get s2_outfit.
        s2_outfit_count = int(num_s2_total * 0.4)

        # First 40% of s2 prompts get s2_outfit
        for i in range(s2_outfit_count):
            p = chosen_s2[i]
            current_outfit = s2_outfit.strip()
            combined = f"{current_outfit}, {p}" if current_outfit else p
            processed = apply_rules_to_prompt(combined, parsed_rules)
            cleaned = cleanup_prompt_tags(processed)
            if cleaned:
                final_prompts_list.append(cleaned)

        # Remaining s2 prompts get s3_outfit
        for i in range(s2_outfit_count, num_s2_total):
            p = chosen_s2[i]
            current_outfit = s3_outfit.strip() # Use s3_outfit here
            combined = f"{current_outfit}, {p}" if current_outfit else p
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
        f.write("s1_prompt_A, blue_sky, 1girl, solo, sweat\n")
        f.write("s1_prompt_B, sunny_day, landscape, mountains, navel\n")
        f.write("s1_prompt_C, forest_path, green_trees, sunlight, 1boy\n")

    with open(os.path.join(sections_dir, "section2.txt"), "w") as f:
        f.write("s2_prompt_1, detailed_face, happy_smile, looking_at_viewer, sweat\n")
        f.write("s2_prompt_2, dynamic_pose, action_shot, blurred_background\n")
        f.write("s2_prompt_3, close_up, thoughtful_expression\n")
        f.write("s2_prompt_4, side_view, walking_fast\n")
        f.write("s2_prompt_5, from_above, sitting_on_ground\n")


    with open(os.path.join(sections_dir, "section3.txt"), "w") as f:
        f.write("s3_prompt_X, cyberpunk_city, neon_lights, rain, 1boy\n")
        f.write("s3_prompt_Y, ancient_ruins, overgrown_vines\n")


    print(f"Testing with base_dir: {script_dir}")

    # Test the core logic function with the new S2 outfit distribution
    # Request 5 prompts for S2 to see the 40/60 split (2 with s2_outfit, 3 with s3_outfit)
    result = generate_prompts_logic(
        s1_count=1, s2_count=5, s3_count=1,
        s1_outfit="STYLE_S1, red_dress",
        s2_outfit="STYLE_S2, green_shirt", # This will be for first 40% of S2
        s3_outfit="STYLE_S3, yellow_hat",  # This will be for last 60% of S2 and all S3
        rules_str="1boy->1man, pale_skin\nsweat->sweat, water_drops, wet\nnavel->\nred_dress->blue_dress",
        seed=42, # Use a fixed seed for predictable sampling
        base_path=script_dir
    )
    print("\n--- Generated Prompts (Logic Test with S2 Split) ---")
    # To make output predictable, let's see which s2 prompts are chosen with seed 42
    # and how they are processed.
    # Assuming section2.txt content above and seed 42, rng.sample(s2_prompts_all, 5) will pick all 5
    # in a specific shuffled order. The split is then applied to this shuffled order.
    # For actual predictable order for this test, let's ensure s2_prompts_all is not shuffled before this test call
    # or manually sort chosen_s2 for this test print if needed (but the function uses rng.sample as intended)

    # To make the test output more clear about which original S2 prompt gets which outfit,
    # we'd need to know the exact order `rng.sample` returns for `chosen_s2` with seed 42.
    # For now, we just observe the count of outfits.
    print(result)
    print("Expected: 1 S1 prompt with STYLE_S1. 2 S2 prompts with STYLE_S2. 3 S2 prompts with STYLE_S3. 1 S3 prompt with STYLE_S3.")
    print("----------------------------------------------------\n")


    # Test the node class (simulating ComfyUI call)
    node_instance = PromptGeneratorNode()
    # Test with s2_count that allows split (e.g., 5)
    output_tuple = node_instance.generate_prompts(
        s1_prompt_count=1,
        s2_prompt_count=5, # To see the split
        s3_prompt_count=1,
        s1_outfit="ultra_detailed_S1, 8k",
        s2_outfit="cinematic_S2",
        s3_outfit="sketchy_S3", # This will be used for some S2 and all S3
        replacement_rules="1girl->1woman\nmountains->snowy_mountains",
        seed=123 # Different seed for different sampling
    )
    print("--- Generated Prompts (Node Test with S2 Split) ---")
    print(output_tuple[0])
    print("Expected: 1 S1 prompt with ultra_detailed_S1. 2 S2 prompts with cinematic_S2. 3 S2 prompts with sketchy_S3. 1 S3 prompt with sketchy_S3.")
    print("-----------------------------------------------\n")

    # Test with fewer S2 prompts than the 40% threshold calculation would naturally split
    # e.g. s2_count = 1. 40% of 1 is 0 (int(0.4)). So it should get s3_outfit.
    # e.g. s2_count = 2. 40% of 2 is 0 (int(0.8)). Both should get s3_outfit.
    # e.g. s2_count = 3. 40% of 3 is 1 (int(1.2)). 1 gets s2_outfit, 2 get s3_outfit.
    result_small_s2 = generate_prompts_logic(
        s1_count=0, s2_count=2, s3_count=0,
        s1_outfit="S1O",
        s2_outfit="S2O_FOR_40_PERCENT",
        s3_outfit="S3O_FOR_60_PERCENT_AND_S3",
        rules_str="", seed=1, base_path=script_dir
    )
    print("--- Generated Prompts (Small S2 Count Test) ---")
    print(f"s2_count=2: {result_small_s2}")
    print("Expected for s2_count=2: Both S2 prompts should get S3O_FOR_60_PERCENT_AND_S3 (because 0.4*2 = 0.8, int(0.8)=0 for s2_outfit_count)")
    print("---------------------------------------------\n")

    result_small_s2_3 = generate_prompts_logic(
        s1_count=0, s2_count=3, s3_count=0,
        s1_outfit="S1O",
        s2_outfit="S2O_FOR_40_PERCENT",
        s3_outfit="S3O_FOR_60_PERCENT_AND_S3",
        rules_str="", seed=1, base_path=script_dir
    )
    print("--- Generated Prompts (Small S2 Count Test) ---")
    print(f"s2_count=3: {result_small_s2_3}")
    print("Expected for s2_count=3: First S2 prompt (from sampled list) gets S2O, next two S2 prompts get S3O (0.4*3=1.2, int(1.2)=1)")
    print("---------------------------------------------\n")

    # Test with more prompts than available (original test)
    result_overflow = generate_prompts_logic(
        s1_count=10, s2_count=1, s3_count=0, # s2_count=1 means it should get s3_outfit
        s1_outfit="overflow_style_S1",
        s2_outfit="overflow_style_S2",
        s3_outfit="overflow_style_S3",
        rules_str="", seed=1, base_path=script_dir
    )
    print("--- Generated Prompts (Overflow Test with S2 Split logic) ---")
    print(result_overflow) # The single s2 prompt should have overflow_style_S3
    print("-----------------------------------------------------------\n")
