import logging
import time
import sys

# Define ANSI escape codes for colors
RESET = "\033[0m"
RED = "\033[31m"


# Custom formatter class
class CustomFormatter(logging.Formatter):
    def __init__(self, start_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = start_time

    def format(self, record):
        # Calculate elapsed time in XXhXXmXXs format
        elapsed_seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time = f"{hours:02}h{minutes:02}m{seconds:02}s"

        # Get module.function name and truncate if too long
        module_name = record.module
        function_name = record.funcName
        full_name = f"{module_name}.{function_name}"
        max_func_length = 40
        if len(full_name) > max_func_length:
            full_name = full_name[:max_func_length - 3] + "..."
        full_name = full_name.ljust(max_func_length)

        # Determine log level prefix
        level_prefix = {
            "DEBUG": "[D]",
            "INFO": "[I]",
            "WARNING": "[W]",
            "ERROR": "[E]",
            "CRITICAL": "[C]",
        }.get(record.levelname, "[ ]")
        color = RED if record.levelname in {"WARNING", "ERROR", "CRITICAL"} else RESET

        # Format the log message
        message = record.getMessage()
        formatted_message = f"{color}{level_prefix} {elapsed_time} {full_name} | {message}{RESET}"
        return formatted_message


def setup_logger():
    start_time = time.time()
    handler = logging.StreamHandler(sys.stdout)
    formatter = CustomFormatter(start_time)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


# Example usage
def example_function_one():
    logger.debug("This is a debug message.")


def very_long_function_name_in_module():
    logger.info("This is an info message with a long function name.")


def warning_example():
    logger.warning("This is a warning message.")


def error_example():
    logger.error("This is an error message.")


if __name__ == "__main__":
    example_function_one()
    very_long_function_name_in_module()
    warning_example()
    error_example()
