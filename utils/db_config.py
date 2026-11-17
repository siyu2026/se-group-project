"""统一数据库配置。

Flask 主服务、爬虫、AI 模块都通过这里获取 MySQL 连接，
连接参数统一从项目根目录的 .env 文件读取（DB_* 前缀）。
"""

import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv

# 项目根目录：utils/ 的上一级，保证从任意工作目录启动都能读到 .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def get_db_config():
    """返回 pymysql 连接参数字典。"""
    return {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'medicalinfo'),
        'charset': os.getenv('DB_CHARSET', 'utf8mb4'),
    }


def get_db_connection(cursor_class=None):
    """获取一个新的数据库连接。

    cursor_class 传入 pymysql.cursors.DictCursor 时可返回字典结果，
    默认返回元组结果（app.py 中按索引取值，需保持默认）。
    """
    kwargs = get_db_config()
    if cursor_class is not None:
        kwargs['cursorclass'] = cursor_class
    return pymysql.connect(**kwargs)


def ensure_database():
    """确保 .env 中配置的数据库存在（不存在则创建），供初始化脚本使用。"""
    kwargs = get_db_config()
    db_name = kwargs.pop('database')
    conn = pymysql.connect(**kwargs)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                'CREATE DATABASE IF NOT EXISTS `{}` DEFAULT CHARACTER SET utf8mb4'.format(db_name)
            )
        conn.commit()
    finally:
        conn.close()
