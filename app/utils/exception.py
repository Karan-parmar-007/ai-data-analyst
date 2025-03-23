import sys

class CustomException(Exception):
    def __init__(self, error, sys_info):
        self.error = error
        self.sys_info = sys_info
        super().__init__(self.__str__())

    def __str__(self):
        return f"CustomException: {self.error} | System Info: {self.sys_info}"
