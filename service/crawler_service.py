"""病例采集服务（单例模式），由 spiders/crawler_scheduler.py 迁入并重构。

【设计模式说明 —— 单例模式（Singleton）】
CrawlerService 使用 SingletonMeta 元类，进程内全局唯一：
    - 数据源清单（由 DataSourceFactory 创建）只需扫描一次；
    - 手动触发（HTTP 接口）与定时触发（APScheduler）共享同一实例，
      保证「采集中去重」「状态一致」。

【工厂模式的使用点】
本类不直接创建 CsvDataSource，而是向 DataSourceFactory 索要数据源实例，
因此数据源类型扩展时本类无需修改。

接口保持与原 CrawlerScheduler 一致：crawl_disease(disease, pages) -> bool，
app.py 的手动触发接口与定时任务都走这里。
"""

from utils.db_config import get_db_connection
from service.base import SingletonMeta
from service.data_source import CSV_FIELDS, DataSourceFactory

# 数据库不可用时使用的兜底疾病列表，与 medicalinfo.sql 内置数据保持一致
DEFAULT_DISEASE_TYPES = ['高血压', '感冒', '腰椎间盘突出', '骨折', '颈椎病', '胃炎', '抑郁症']

INSERT_SQL = (
    'INSERT INTO cases (type, gender, age, `time`, content, docName, docHospital, '
    'department, detailUrl, height, weight, illDuration, allergy) '
    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
)


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


class CrawlerService(metaclass=SingletonMeta):
    """按疾病执行采集任务，并把新病例写入 cases 表（单例）。"""

    def __init__(self):
        # 【工厂模式】通过工厂创建数据源实例，而非自行 new 具体类
        self.data_sources = DataSourceFactory.discover_default_sources()

    def _fetch_rows(self, disease):
        """遍历所有数据源，收集指定疾病的病例行。"""
        rows = []
        for source in self.data_sources:
            rows.extend(source.fetch(disease))
        return rows

    @staticmethod
    def _existing_keys(cursor, disease):
        cursor.execute(
            'SELECT detailUrl FROM cases WHERE type=%s AND detailUrl IS NOT NULL AND detailUrl <> ""',
            (disease,)
        )
        return {row[0] for row in cursor.fetchall()}

    def crawl_disease(self, disease, pages=3):
        """采集指定疾病的新病例。

        pages 参数保留以兼容历史调用，本地数据源模式下不生效。
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

            # 增量导入：按 detailUrl 去重，已存在的链接跳过
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


# 模块级单例实例，保持与原 crawler_scheduler 相同的使用方式
crawler_service = CrawlerService()
