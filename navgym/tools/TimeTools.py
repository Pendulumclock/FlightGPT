from datetime import datetime


def time_str():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")
