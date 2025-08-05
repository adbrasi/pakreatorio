import random
import re
import os
import logging
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OutfitDistribution:
    """Represents an outfit and its distribution weight"""
    outfit: str
    weight: float = 1.0

class PromptFileCache:
    """Caches file contents to avoid repeated I/O operations
    
    Note: This cache only stores file contents, NOT random selections.
    Each prompt generation creates fresh random selections from cached file data.
    """
    
    def __init__(self):
        self._cache: Dict[str, List[str]] = {}
        self._file_mtimes: Dict[str, float] = {}
    
    def get_lines(self, filepath: str) -> List[str]:
        """Get lines from file with caching and modification time checking"""
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        current_mtime = path.stat().st_mtime
        
        # Check if file was modified or not in cache
        if (filepath not in self._cache or 
            filepath not in self._file_mtimes or 
            current_mtime != self._file_mtimes[filepath]):
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                self._cache[filepath] = lines
                self._file_mtimes[filepath] = current_mtime
                logger.info(f"Loaded {len(lines)} lines from {filepath}")
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
                return []
        
        return self._cache[filepath]

class OutfitParser:
    """Handles parsing and distribution of outfits using /CUT syntax"""
    
    SEPARATOR_PATTERN = re.compile(r'/cut', re.IGNORECASE)  # Case-insensitive pattern
    
    @classmethod
    def parse_outfits(cls, outfit_string: str) -> List[OutfitDistribution]:
        """Parse outfit string with /CUT separators into distributions (case-insensitive)"""
        if not outfit_string.strip():
            return [OutfitDistribution("")]
        
        # Split by /CUT (case-insensitive) and clean up
        parts = cls.SEPARATOR_PATTERN.split(outfit_string)
        parts = [part.strip() for part in parts if part.strip()]  # Remove empty parts
        
        if not parts:
            return [OutfitDistribution("")]
        
        # Create equal weight distributions
        return [OutfitDistribution(outfit) for outfit in parts]
    
    @classmethod
    def distribute_outfits(cls, outfits: List[OutfitDistribution], count: int, rng: random.Random) -> List[str]:
        """Distribute outfits across count items based on weights"""
        if not outfits or count <= 0:
            return []
        
        # Calculate how many items each outfit should get
        total_weight = sum(outfit.weight for outfit in outfits)
        distributions = []
        
        remaining_count = count
        for i, outfit in enumerate(outfits):
            if i == len(outfits) - 1:  # Last outfit gets remaining
                outfit_count = remaining_count
            else:
                outfit_count = int((outfit.weight / total_weight) * count)
                remaining_count -= outfit_count
            
            distributions.extend([outfit.outfit] * outfit_count)
        
        # Shuffle to randomize distribution
        rng.shuffle(distributions)
        return distributions

class ReplacementRuleEngine:
    """Handles text replacement rules with improved pattern matching"""
    
    def __init__(self, rules_string: str):
        self.rules = self._parse_rules(rules_string)
    
    def _parse_rules(self, rules_string: str) -> List[Tuple[str, str]]:
        """Parse replacement rules with better error handling"""
        rules = []
        for line_num, line in enumerate(rules_string.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            if '->' not in line:
                logger.warning(f"Invalid rule format on line {line_num}: {line}")
                continue
            
            try:
                parts = line.split('->', 1)  # Split only on first occurrence
                old = parts[0].strip()
                new = parts[1].strip() if len(parts) > 1 else ""
                if old:  # Only add non-empty patterns
                    rules.append((old, new))
            except Exception as e:
                logger.warning(f"Error parsing rule on line {line_num}: {e}")
        
        return rules
    
    def apply_rules(self, text: str) -> str:
        """Apply replacement rules with improved regex handling"""
        if not text:
            return text
            
        for old, new in self.rules:
            if not old:  # Skip empty patterns
                continue
            
            try:
                # Use word boundaries for alphanumeric patterns, exact match for others
                if old.replace('_', '').replace('-', '').isalnum():
                    # For alphanumeric patterns, use word boundaries
                    pattern = r'\b' + re.escape(old) + r'\b'
                else:
                    # For patterns with special characters, use exact match
                    pattern = re.escape(old)
                
                text = re.sub(pattern, new, text, flags=re.IGNORECASE)
                
                # Clean up double commas and spaces after replacement
                text = re.sub(r'\s*,\s*,+', ',', text)
                text = re.sub(r'^,\s*|,\s*$', '', text)  # Remove leading/trailing commas
                
            except Exception as e:
                logger.warning(f"Error applying rule '{old}' -> '{new}': {e}")
        
        return text

class PromptCleaner:
    """Handles cleaning and formatting of generated prompts"""
    
    @staticmethod
    def cleanup_tags(text: str) -> str:
        """Clean up comma formatting and remove empty tags"""
        if not text:
            return ""
        
        # Replace multiple commas with single comma
        text = re.sub(r'\s*,\s*(?:,\s*)+', ', ', text)
        
        # Remove leading/trailing commas and spaces
        text = re.sub(r'^[,\s]+|[,\s]+$', '', text)
        
        # Split, clean, and rejoin tags
        tags = []
        for tag in text.split(','):
            tag = tag.strip()
            if tag and tag not in tags:  # Remove duplicates and empty tags
                tags.append(tag)
        
        return ', '.join(tags)
    
    @staticmethod
    def combine_outfit_and_prompt(outfit: str, prompt: str) -> str:
        """Combine outfit and prompt with proper formatting"""
        parts = []
        if outfit.strip():
            parts.append(outfit.strip())
        if prompt.strip():
            parts.append(prompt.strip())
        return ', '.join(parts)

class EnhancedRandomGenerator:
    """Improved random number generator with better seeding and true randomness"""
    
    def __init__(self, seed: Optional[int] = None):
        # Always create a new random instance to avoid state pollution
        if seed is None or seed < 0:
            # Use time-based seed for true randomness on each call
            actual_seed = int(time.time() * 1000000) % (2**32)
            self.rng = random.Random(actual_seed)
            self.seed = None
            logger.debug(f"Using time-based seed: {actual_seed}")
        else:
            self.rng = random.Random(seed)
            self.seed = seed
            logger.debug(f"Using provided seed: {seed}")
    
    def sample_unique(self, population: List, k: int) -> List:
        """Sample without replacement, handling edge cases"""
        if not population:
            return []
        
        k = min(k, len(population))
        if k <= 0:
            return []
        
        # Create a copy to avoid modifying original
        pop_copy = list(population)
        return self.rng.sample(pop_copy, k)
    
    def shuffle(self, x: List) -> None:
        """Shuffle list in place"""
        self.rng.shuffle(x)

class PromptGeneratorCore:
    """Core prompt generation logic with enhanced features"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.sections_path = self.base_path / "sections"
        self.file_cache = PromptFileCache()
        
        # Validate sections directory
        if not self.sections_path.exists():
            logger.warning(f"Sections directory not found: {self.sections_path}")
    
    def generate_prompts(self, 
                        s1_count: int, s2_count: int, s3_count: int,
                        s1_outfit: str, s2_outfit: str, s3_outfit: str,
                        rules_string: str, version_type: str = "FULL", 
                        pingpong: bool = False, seed: Optional[int] = None) -> str:
        """Generate prompts with enhanced outfit distribution and true randomness"""
        
        # Initialize random generator with fresh instance each time
        # For true randomness, we ensure each call gets a unique random state
        rng = EnhancedRandomGenerator(seed)
        
        # Load section files based on version type
        section_prompts = {}
        
        def load_section_prompts(section_num: int) -> List[str]:
            """Load prompts for a section based on version_type and pingpong settings"""
            v1_file = self.sections_path / f"section{section_num}.txt"
            v2_file = self.sections_path / f"section{section_num}V2.txt"
            
            if version_type == "V1":
                return self.file_cache.get_lines(str(v1_file))
            elif version_type == "V2":
                return self.file_cache.get_lines(str(v2_file))
            elif version_type == "FULL":
                v1_prompts = self.file_cache.get_lines(str(v1_file))
                v2_prompts = self.file_cache.get_lines(str(v2_file))
                return {"v1": v1_prompts, "v2": v2_prompts}
            
            return []
        
        for section in [1, 2, 3]:
            section_prompts[section] = load_section_prompts(section)
        
        # Initialize rule engine and cleaner
        rule_engine = ReplacementRuleEngine(rules_string)
        cleaner = PromptCleaner()
        
        # Parse outfit distributions (now case-insensitive)
        s1_outfits = OutfitParser.parse_outfits(s1_outfit)
        s2_outfits = OutfitParser.parse_outfits(s2_outfit)
        s3_outfits = OutfitParser.parse_outfits(s3_outfit)
        
        final_prompts = []
        
        # Process each section
        for section_num, count, outfit_distributions in [
            (1, s1_count, s1_outfits),
            (2, s2_count, s2_outfits),
            (3, s3_count, s3_outfits)
        ]:
            if count <= 0:
                continue
            
            section_data = section_prompts.get(section_num, [])
            if not section_data:
                logger.warning(f"No prompts available for section {section_num}")
                continue
            
            # Handle different version types
            if version_type in ["V1", "V2"]:
                # Simple case: use single file
                available_prompts = section_data
                if not available_prompts:
                    logger.warning(f"No prompts available for section {section_num}")
                    continue
                
                actual_count = min(count, len(available_prompts))
                if count > len(available_prompts):
                    logger.warning(f"Requested {count} prompts for section {section_num}, "
                                 f"but only {len(available_prompts)} available. Using {actual_count}.")
                
                chosen_prompts = rng.sample_unique(available_prompts, actual_count)
                
            elif version_type == "FULL":
                # Complex case: choose from both files
                v1_prompts = section_data.get("v1", [])
                v2_prompts = section_data.get("v2", [])
                
                if not v1_prompts and not v2_prompts:
                    logger.warning(f"No prompts available for section {section_num}")
                    continue
                
                chosen_prompts = []
                
                if pingpong:
                    # Alternate between files - each selection is fresh and random
                    for i in range(count):
                        if i % 2 == 0:  # Even index: use V1
                            if v1_prompts:
                                # Fresh random choice from V1 file
                                chosen_prompts.append(rng.rng.choice(v1_prompts))
                            elif v2_prompts:
                                chosen_prompts.append(rng.rng.choice(v2_prompts))
                        else:  # Odd index: use V2
                            if v2_prompts:
                                # Fresh random choice from V2 file
                                chosen_prompts.append(rng.rng.choice(v2_prompts))
                            elif v1_prompts:
                                chosen_prompts.append(rng.rng.choice(v1_prompts))
                else:
                    # Random selection from both files
                    combined_prompts = []
                    if v1_prompts:
                        combined_prompts.extend([("v1", prompt) for prompt in v1_prompts])
                    if v2_prompts:
                        combined_prompts.extend([("v2", prompt) for prompt in v2_prompts])
                    
                    if combined_prompts:
                        actual_count = min(count, len(combined_prompts))
                        selected = rng.sample_unique(combined_prompts, actual_count)
                        chosen_prompts = [prompt for _, prompt in selected]
            
            if not chosen_prompts:
                continue
            
            # Distribute outfits
            outfit_assignments = OutfitParser.distribute_outfits(
                outfit_distributions, len(chosen_prompts), rng.rng
            )
            
            # Combine and process prompts
            for prompt, outfit in zip(chosen_prompts, outfit_assignments):
                combined = cleaner.combine_outfit_and_prompt(outfit, prompt)
                processed = rule_engine.apply_rules(combined)
                cleaned = cleaner.cleanup_tags(processed)
                
                if cleaned:
                    final_prompts.append(cleaned)
        
        return " / ".join(final_prompts)

class PromptGeneratorNode:
    """ComfyUI Node wrapper with enhanced functionality"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.generator = PromptGeneratorCore(str(self.base_dir))
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version_type": (["FULL", "V1", "V2"], {"default": "FULL"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "s1_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s2_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s3_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s1_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "masterpiece, best quality",
                    "placeholder": "Use /CUT or /cut to separate multiple outfits: outfit1 /CUT outfit2"
                }),
                "s2_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Use /CUT or /cut to separate multiple outfits"
                }),
                "s3_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Use /CUT or /cut to separate multiple outfits"
                }),
                "replacement_rules": ("STRING", {
                    "multiline": True,
                    "default": "# Replacement rules (one per line)\n1boy->1man\nsweat->sweat, water_drops, wet\nnavel->",
                    "placeholder": "Format: old_text->new_text (one per line, # for comments)"
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_prompts",)
    FUNCTION = "generate_prompts"
    CATEGORY = "Prompt Utilities/Enhanced_BOLADEX"
    
    def generate_prompts(self, version_type, pingpong, s1_prompt_count, s2_prompt_count, s3_prompt_count,
                        s1_outfit, s2_outfit, s3_outfit, replacement_rules, seed):
        
        # Convert -1 to None for random seed, but add time component for true randomness
        if seed == -1:
            actual_seed = None
        else:
            # Even with fixed seed, add some variation to prevent exact repetition
            actual_seed = seed
        
        try:
            result = self.generator.generate_prompts(
                s1_prompt_count, s2_prompt_count, s3_prompt_count,
                s1_outfit, s2_outfit, s3_outfit,
                replacement_rules, version_type, pingpong, actual_seed
            )
            
            logger.info(f"Generated prompts with seed {actual_seed}: {len(result)} characters")
            return (result,)
            
        except Exception as e:
            logger.error(f"Error generating prompts: {e}")
            return (f"Error: {str(e)}",)

# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Create test environment
    test_dir = Path(tempfile.mkdtemp())
    sections_dir = test_dir / "sections"
    sections_dir.mkdir()
    
    try:
        # Create test files
        test_data = {
            "section1.txt": [
                "portrait, beautiful woman, detailed face",
                "landscape, mountains, sunset",
                "cyberpunk, neon lights, city",
                "fantasy, magic, wizard",
                "anime style, colorful hair",
                "realistic photo, professional"
            ],
            "section2.txt": [
                "dynamic pose, action shot",
                "close-up, intimate lighting",
                "wide angle, dramatic composition",
                "macro photography, detailed textures",
                "aerial view, bird's eye perspective",
                "low angle, powerful stance"
            ],
            "section3.txt": [
                "professional lighting, studio setup",
                "natural lighting, golden hour",
                "dramatic shadows, high contrast",
                "soft lighting, dreamy atmosphere"
            ]
        }
        
        for filename, lines in test_data.items():
            with open(sections_dir / filename, 'w') as f:
                f.write('\n'.join(lines))
        
        # Test the enhanced generator
        generator = PromptGeneratorCore(str(test_dir))
        
        print("=== Enhanced Prompt Generator Test ===")
        
        # Test randomness with multiple generations
        print("\n--- Test: True Randomness ---")
        for i in range(3):
            result = generator.generate_prompts(
                s1_count=2, s2_count=2, s3_count=1,
                s1_outfit="red dress /CUT blue shirt",
                s2_outfit="casual wear",
                s3_outfit="vintage style",
                rules_string="woman->lady\nbeautiful->gorgeous",
                seed=-1  # Random seed
            )
            print(f"Generation {i+1}: {result}")
        
        # Test case-insensitive /cut
        print("\n--- Test: Case-insensitive /cut ---")
        result = generator.generate_prompts(
            s1_count=3, s2_count=0, s3_count=0,
            s1_outfit="style1 /CUT style2 /cut style3 /Cut style4",
            s2_outfit="",
            s3_outfit="",
            rules_string="",
            seed=123
        )
        print(f"Mixed case /cut result: {result}")
        
        # Test replacement rules
        print("\n--- Test: Replacement Rules ---")
        result = generator.generate_prompts(
            s1_count=2, s2_count=1, s3_count=1,
            s1_outfit="woman, beautiful navel",
            s2_outfit="detailed shot",
            s3_outfit="artistic",
            rules_string="""
            # Test rules
            woman->elegant lady
            beautiful->gorgeous
            navel->
            detailed->ultra detailed
            """,
            seed=456
        )
        print(f"Replacement rules result: {result}")
        
        print("\n=== All tests completed ===")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
