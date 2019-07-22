# -*- coding: utf-8 -*-
# @Time     : 2019/7/19 18:23
# @Author   : buracagyang
# @File     : log_config.py
# @Software : PyCharm

"""
Describe:
        
"""
import os
import logging.config
try:
    import yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml


def setup_logging(default_path="logger.yaml", default_level=logging.INFO, env_key="LOG_CFG"):

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
            config["handlers"]["info_file_handler"]["filename"] = os.path.join(base_dir, "info.log")
            config["handlers"]["error_file_handler"]["filename"] = os.path.join(base_dir, "errors.log")
            logging.config.dictConfig(config)

    else:
        logging.basicConfig(level=default_level,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='logging.log',
                            filemode='w')


if __name__ == "__main__":
    setup_logging('logger.yaml')
    logging.info('test')
