"""settings and configuration for retina net

Author: baiyu
"""

from config import global_settings

class Settings:
    def __init__(self, settings_module):

        for settings in dir(settings_module):
            if settings.isupper():
                setattr(self, settings, getattr(settings_module, settings))

settings = Settings(global_settings)
