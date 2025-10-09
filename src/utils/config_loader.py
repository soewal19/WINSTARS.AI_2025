import json
from pathlib import Path
class Config:
    def __init__(self, path='config/project_config.json'):
        self.path = Path(path)
        self._cfg = json.loads(self.path.read_text()) if self.path.exists() else {}
    def get(self, *keys, default=None):
        data = self._cfg
        for k in keys:
            if not isinstance(data, dict): return default
            data = data.get(k, default)
        return data
