from datetime import datetime


def time_str():
    """
    获取当前时间的字符串表示
    :return: 当前时间的字符串表示
    """
    return datetime.now().strftime("%Y%m%d%H%M%S%f")