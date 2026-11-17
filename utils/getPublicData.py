"""通用查询与格式化工具。

getAllData.py 中的统计函数都会复用这里的查询封装，避免重复建立连接。
"""

import re
from datetime import datetime

from utils.db_config import get_db_connection


def getSqlData(sql, params=None):
    """执行查询并返回全部结果（元组列表）。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def executeSql(sql, params=None):
    """执行写入语句并返回受影响行数。"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        affected = cursor.execute(sql, params or ())
        conn.commit()
        return affected
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def getNowFormatDate():
    """返回当前时间字符串，供报告/日志使用。"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def safe_int(value, default=0):
    """把数据库中 varchar 类型的数字（如 '35'、'163'）安全转换为 int。"""
    if value is None:
        return default
    match = re.search(r'-?\d+(\.\d+)?', str(value))
    if not match:
        return default
    return int(float(match.group(0)))


def safe_float(value, default=0.0):
    """把 varchar 类型的数值安全转换为 float。"""
    if value is None:
        return default
    match = re.search(r'-?\d+(\.\d+)?', str(value))
    if not match:
        return default
    return float(match.group(0))
