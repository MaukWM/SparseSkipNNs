from enum import Enum


class LogLevel(Enum):

    VERBOSE = 0
    SIMPLE = 1
    NONE = 2

    def __le__(self, b):
        return self.value <= b.value
