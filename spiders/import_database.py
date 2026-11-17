"""初始化数据库：创建 medicalinfo 库并导入 medicalinfo.sql。

用法（在项目根目录）：
    python -m spiders.import_database
"""

import re
from pathlib import Path

from utils.db_config import ensure_database, get_db_config, get_db_connection

BASE_DIR = Path(__file__).resolve().parent.parent
SQL_FILE = BASE_DIR / 'medicalinfo.sql'


def _split_statements(sql_text):
    """按分号切分 SQL 语句，忽略注释与引号内的分号。"""
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.S)
    sql_text = re.sub(r'^\s*--.*$', '', sql_text, flags=re.M)

    statements = []
    buffer = []
    quote = None
    for char in sql_text:
        if quote:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            buffer.append(char)
        elif char == ';':
            statement = ''.join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)

    tail = ''.join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def import_database():
    if not SQL_FILE.exists():
        raise FileNotFoundError('未找到 {}'.format(SQL_FILE))

    ensure_database()
    print('[DB] 目标数据库：{}'.format(get_db_config()['database']))

    sql_text = SQL_FILE.read_text(encoding='utf-8', errors='ignore')
    statements = _split_statements(sql_text)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for statement in statements:
            cursor.execute(statement)
        conn.commit()
        cursor.execute('SELECT COUNT(*) FROM cases')
        total = cursor.fetchone()[0]
        cursor.close()
        print('[DB] 导入完成，共执行 {} 条语句，cases 表 {} 条记录'.format(len(statements), total))
    finally:
        conn.close()


if __name__ == '__main__':
    import_database()
