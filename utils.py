# Global debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def set_debug(value: bool):
    """Set the debug state"""
    global DEBUG
    DEBUG = value 