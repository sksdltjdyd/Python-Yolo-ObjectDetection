from enum import Enum

class AppMode(Enum):
    SETUP = "SETUP"
    RUNNING = "RUNNING"

class SetupStep(Enum):
    MASK_AREA = 1
    BACKGROUND = 2
    CALIBRATE = 3

class StateManager:
    """애플리케이션의 전반적인 상태를 관리"""
    def __init__(self):
        self.app_mode = AppMode.SETUP
        self.setup_step = SetupStep.MASK_AREA