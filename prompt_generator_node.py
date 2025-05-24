import random
import re
import os
import logging
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
    """Caches file contents to avoid repeated I/O operations"""
    
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
    
    SEPARATOR = "/CUT"
    
    @classmethod
    def parse_outfits(cls, outfit_string: str) -> List[OutfitDistribution]:
        """Parse outfit string with /CUT separators into distributions"""
        if not outfit_string.strip():
            return [OutfitDistribution("")]
        
        # Split by /CUT and clean up
        parts = [part.strip() for part in outfit_string.split(cls.SEPARATOR)]
        parts = [part for part in parts if part]  # Remove empty parts
        
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
                old, new = line.split('->', 1)
                rules.append((old.strip(), new.strip()))
            except Exception as e:
                logger.warning(f"Error parsing rule on line {line_num}: {e}")
        
        return rules
    
    def apply_rules(self, text: str) -> str:
        """Apply replacement rules with improved regex handling"""
        for old, new in self.rules:
            if not old:  # Skip empty patterns
                continue
            
            try:
                # Use word boundaries for better matching, but handle special cases
                if old.isalnum():
                    pattern = r'\b' + re.escape(old) + r'\b'
                else:
                    pattern = re.escape(old)
                
                text = re.sub(pattern, new, text, flags=re.IGNORECASE)
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
        text = re.sub(r'\s*,\s*(?:,\s*)+', ',', text)
        
        # Split, clean, and rejoin tags
        tags = []
        for tag in text.split(','):
            tag = tag.strip()
            if tag and tag not in tags:  # Remove duplicates
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
    """Improved random number generator with better seeding"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is None or seed < 0:
            # Use system random for true randomness
            self.rng = random.SystemRandom()
            self.seed = None
        else:
            self.rng = random.Random(seed)
            self.seed = seed
    
    def sample_unique(self, population: List, k: int) -> List:
        """Sample without replacement, handling edge cases"""
        if not population:
            return []
        
        k = min(k, len(population))
        if k <= 0:
            return []
        
        return self.rng.sample(population, k)
    
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
                        rules_string: str, seed: Optional[int] = None) -> str:
        """Generate prompts with enhanced outfit distribution"""
        
        # Initialize random generator
        rng = EnhancedRandomGenerator(seed)
        
        # Load section files
        section_files = {
            1: self.sections_path / "section1.txt",
            2: self.sections_path / "section2.txt", 
            3: self.sections_path / "section3.txt"
        }
        
        section_prompts = {}
        for section, filepath in section_files.items():
            section_prompts[section] = self.file_cache.get_lines(str(filepath))
        
        # Initialize rule engine and cleaner
        rule_engine = ReplacementRuleEngine(rules_string)
        cleaner = PromptCleaner()
        
        # Parse outfit distributions
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
            
            available_prompts = section_prompts.get(section_num, [])
            if not available_prompts:
                logger.warning(f"No prompts available for section {section_num}")
                continue
            
            # Sample prompts
            actual_count = min(count, len(available_prompts))
            if count > len(available_prompts):
                logger.warning(f"Requested {count} prompts for section {section_num}, "
                             f"but only {len(available_prompts)} available. Using {actual_count}.")
            
            chosen_prompts = rng.sample_unique(available_prompts, actual_count)
            
            # Distribute outfits
            outfit_assignments = OutfitParser.distribute_outfits(
                outfit_distributions, actual_count, rng.rng
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
                "s1_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s2_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s3_prompt_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "s1_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "masterpiece, best quality",
                    "placeholder": "Use /CUT to separate multiple outfits: outfit1 /CUT outfit2"
                }),
                "s2_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Use /CUT to separate multiple outfits"
                }),
                "s3_outfit": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Use /CUT to separate multiple outfits"
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
    
    def generate_prompts(self, s1_prompt_count, s2_prompt_count, s3_prompt_count,
                        s1_outfit, s2_outfit, s3_outfit, replacement_rules, seed):
        
        # Convert -1 to None for random seed
        actual_seed = None if seed == -1 else seed
        
        try:
            result = self.generator.generate_prompts(
                s1_prompt_count, s2_prompt_count, s3_prompt_count,
                s1_outfit, s2_outfit, s3_outfit,
                replacement_rules, actual_seed
            )
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
                "fantasy, magic, wizard"
            ],
            "section2.txt": [
                "dynamic pose, action shot",
                "close-up, intimate lighting",
                "wide angle, dramatic composition",
                "macro photography, detailed textures",
                "aerial view, bird's eye perspective"
            ],
            "section3.txt": [
                "professional lighting, studio setup",
                "natural lighting, golden hour",
                "dramatic shadows, high contrast"
            ]
        }
        
        for filename, lines in test_data.items():
            with open(sections_dir / filename, 'w') as f:
                f.write('\n'.join(lines))
        
        # Test the enhanced generator
        generator = PromptGeneratorCore(str(test_dir))
        
        print("=== Enhanced Prompt Generator Test ===")
        
        # Test 1: Multiple outfits with /CUT
        print("\n--- Test 1: Multiple Outfits ---")
        result1 = generator.generate_prompts(
            s1_count=4, s2_count=4, s3_count=2,
            s1_outfit="red dress /CUT blue shirt /CUT green jacket",
            s2_outfit="casual wear /CUT formal attire",
            s3_outfit="vintage style",
            rules_string="woman->lady\nbeautiful->gorgeous",
            seed=42
        )
        print(f"Result: {result1}")
        
        # Test 2: Empty outfit handling
        print("\n--- Test 2: Empty Outfits ---")
        result2 = generator.generate_prompts(
            s1_count=2, s2_count=0, s3_count=1,
            s1_outfit="",
            s2_outfit="should not appear",
            s3_outfit="only style",
            rules_string="",
            seed=123
        )
        print(f"Result: {result2}")
        
        # Test 3: Complex rules
        print("\n--- Test 3: Complex Rules ---")
        result3 = generator.generate_prompts(
            s1_count=2, s2_count=2, s3_count=1,
            s1_outfit="masterpiece /CUT high quality",
            s2_outfit="detailed",
            s3_outfit="artistic",
            rules_string="""
            # Character replacements
            woman->elegant woman
            lighting->professional lighting
            # Remove unwanted terms
            detailed->
            """,
            seed=456
        )
        print(f"Result: {result3}")
        
        print("\n=== All tests completed ===")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
