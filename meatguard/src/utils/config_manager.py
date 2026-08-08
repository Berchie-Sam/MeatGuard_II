import yaml
import os
from pathlib import Path

class ConfigManager:
    """
    Handles hierarchical YAML loading for MeatGuard II.
    Resolves the master config.yaml into a unified configuration object.
    """
    def __init__(self, config_path="configs/config.yaml"):
        self.base_path = Path(config_path).parent
        self.config = self._load_yaml(config_path)
        self._resolve_pointers()

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _resolve_pointers(self):
        """
        Resolves pointers in the master config to specialized YAML files.
        """
        # Resolve Data
        data_ptr = self.config.get('data')
        if data_ptr:
            data_path = self.base_path / "data" / f"{data_ptr}.yaml"
            self.config['data_details'] = self._load_yaml(data_path)

        # Resolve Architecture
        arch_ptr = self.config.get('architecture')
        if arch_ptr:
            arch_path = self.base_path / "model" / f"{arch_ptr}.yaml"
            self.config['arch_details'] = self._load_yaml(arch_path)

        # Resolve Augmentation
        aug_ptr = self.config.get('augment')
        if aug_ptr:
            aug_path = self.base_path / "train" / "augment" / f"{aug_ptr}.yaml"
            self.config['aug_details'] = self._load_yaml(aug_path)

        # Resolve Viz Style
        style_ptr = self.config.get('visualization', {}).get('style_preset')
        if style_ptr:
            style_path = self.base_path / "viz" / "styles" / f"{style_ptr}.yaml"
            self.config['viz_style'] = self._load_yaml(style_path)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]
