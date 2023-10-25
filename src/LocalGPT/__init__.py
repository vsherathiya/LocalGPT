# ====================================================================================
# ==========================================Exception=================================
# ====================================================================================

from datetime import datetime
import sys
import os

import logging

import traceback

def error_message_detail(error):
    """
    Construct an error message with detailed information about the error.

    :param error: The error message that was raised
    :type error: Exception
    :return: The formatted error message
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_message = f"""Error Type: {exc_type.__name__}
                File: {exc_traceback.tb_frame.f_code.co_filename}
                Line (Try Block): {exc_traceback.tb_lineno}
                Line (Exception Block): {exc_traceback.tb_frame.f_lineno}
                Error Message: {str(error)}
                """
    return error_message

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message  # You don't need to use error_message_detail here

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.error_message}')"

# =============================================================================
# =======================================logger================================
# =============================================================================


logging_string = "[%(asctime)s] || %(filename)s || %(lineno)d || %(name)s || %(funcName)s() || %(lineno)s || %(levelname)s || %(message)s"

log_dir = 'logs'


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y%m%d%H')}"


log_file_path = os.path.join(
    log_dir, get_current_time_stamp(), 'running_logs.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format=logging_string,
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger()
