from .prompt_generator_node import PromptGeneratorNode

NODE_CLASS_MAPPINGS = {
    "PromptGenerator_PackCREATOR_BOLADEX": PromptGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator_PackCREATOR_BOLADEX": "Prompt Generator (Pack Boladex)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("------------------------------------------")
print("PackCREATOR_BOLADEX Nodes LOADED")
print("------------------------------------------")