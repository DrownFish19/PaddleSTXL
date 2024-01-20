import logging


class Logger:
    def __init__(self, name, log_file):
        self.name = name
        self.log_file = log_file
        self.logger = self._get_logger()

    def _get_logger(self):
        # 创建一个logger
        logger = logging.getLogger(self.name)
        # 设置logger的日志级别
        logger.setLevel(logging.INFO)
        # 创建一个handler，用于写入日志文件
        file_handler = logging.FileHandler(self.log_file)
        # 定义handler的输出格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        # 给logger添加handler
        logger.addHandler(file_handler)
        return logger

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
