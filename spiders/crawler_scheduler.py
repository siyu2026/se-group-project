"""爬虫调度模块。

app.py 通过 `from crawler_scheduler import crawler_scheduler, disease_types` 使用本模块。

说明：原始版本使用 selenium 实时抓取好大夫在线，依赖外网且容易被反爬阻断。
这里改为「本地数据源增量导入」：从项目根目录的 temp2.csv 以及
spiders/data_source/*.csv 读取病例，按 detailUrl 去重后写入 MySQL。
接口保持不变（crawl_disease 返回 bool），后续要接真实爬虫时替换 _fetch_rows 即可。
"""

import csv
import glob
import os
from pathlib import Path

from utils.db_config import get_db_connection

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SOURCE_DIR = Path(__file__).resolve().parent / 'data_source'

# 数据库不可用时使用的兜底疾病列表，与 medicalinfo.sql 内置数据保持一致
DEFAULT_DISEASE_TYPES = ['高血压', '感冒', '腰椎间盘突出', '骨折', '颈椎病', '胃炎', '抑郁症']

INSERT_SQL = (
    'INSERT INTO cases (type, gender, age, `time`, content, docName, docHospital, '
    'department, detailUrl, height, weight, illDuration, allergy) '
    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
)

CSV_FIELDS = [
    'type', 'gender', 'age', 'time', 'content', 'docName', 'docHospital',
    'department', 'detailUrl', 'height', 'weight', 'illDuration', 'allergy',
]


def _load_disease_types():
    """优先从 cases 表读取疾病类型，失败时回退到内置列表。"""
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT DISTINCT type FROM cases WHERE type IS NOT NULL AND type <> "" ORDER BY type'
                )
                rows = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        return rows or list(DEFAULT_DISEASE_TYPES)
    except Exception as e:
        print('[Crawler] 读取疾病类型失败，使用默认列表：{}'.format(e))
        return list(DEFAULT_DISEASE_TYPES)


# app.py 依赖的疾病列表
disease_types = _load_disease_types()


class CrawlerScheduler:
    """按疾病执行采集任务，并把新病例写入 cases 表。"""

    def __init__(self):
        self.data_source_files = self._discover_files()

    @staticmethod
    def _discover_files():
        files = []
        csv_default = BASE_DIR / 'temp2.csv'
        if csv_default.exists():
            files.append(csv_default)
        files.extend(Path(p) for p in glob.glob(str(DATA_SOURCE_DIR / '*.csv')))
        return files

    def _fetch_rows(self, disease):
        """从本地 CSV 数据源读取指定疾病的病例行。"""
        rows = []
        for path in self.data_source_files:
            try:
                with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                    for item in csv.DictReader(f):
                        if (item.get('type') or '').strip() != disease:
                            continue
                        rows.append(tuple((item.get(field) or '').strip() or None for field in CSV_FIELDS))
            except Exception as e:
                print('[Crawler] 读取数据源 {} 失败：{}'.format(path, e))
        return rows

    def _existing_keys(self, cursor, disease):
        cursor.execute(
            'SELECT detailUrl FROM cases WHERE type=%s AND detailUrl IS NOT NULL AND detailUrl <> ""',
            (disease,)
        )
        return {row[0] for row in cursor.fetchall()}

    def crawl_disease(self, disease, pages=3):
        """采集指定疾病的新病例。

        pages 参数保留以兼容 app.py 的调用，本地数据源模式下不生效。
        返回 True 表示任务执行完成（新增条数可能为 0）。
        """
        rows = self._fetch_rows(disease)
        if not rows:
            print('[Crawler] 数据源中没有 {} 的记录'.format(disease))
            return True

        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            existing = self._existing_keys(cursor, disease)

            new_rows = []
            for row in rows:
                detail_url = row[CSV_FIELDS.index('detailUrl')]
                if detail_url and detail_url in existing:
                    continue
                if detail_url:
                    existing.add(detail_url)
                new_rows.append(row)

            if new_rows:
                cursor.executemany(INSERT_SQL, new_rows)
                conn.commit()
            print('[Crawler] {} 采集完成，新增 {} 条'.format(disease, len(new_rows)))
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            print('[Crawler] {} 采集失败：{}'.format(disease, e))
            raise
        finally:
            if conn:
                conn.close()


crawler_scheduler = CrawlerScheduler()
