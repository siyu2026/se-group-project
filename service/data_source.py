"""数据源抽象与工厂（工厂模式）。

【设计模式说明 —— 工厂模式（Factory）】
目的：把「对象的创建」与「对象的使用」解耦。
CrawlerService 只面向 DataSource 抽象接口编程（调用 fetch），
并不关心数据来自本地 CSV、真实网页爬虫还是第三方 API。
新增数据源类型时，只需：
    1. 新增一个 DataSource 子类（如 WebCrawlerDataSource）；
    2. 在 DataSourceFactory 注册一行：DataSourceFactory.register('web', WebCrawlerDataSource)；
调用方（CrawlerService）零改动 —— 这就是开闭原则（对扩展开放，对修改关闭）。

当前默认实现为 CsvDataSource：从本地 CSV 增量读取病例，
替代原 selenium 实时抓取方案（依赖外网且易被反爬阻断）。
"""

import csv
import glob
from abc import ABC, abstractmethod
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SOURCE_DIR = BASE_DIR / 'spiders' / 'data_source'

# cases 表字段顺序（不含自增 id），与 INSERT_SQL 的列顺序保持一致
CSV_FIELDS = [
    'type', 'gender', 'age', 'time', 'content', 'docName', 'docHospital',
    'department', 'detailUrl', 'height', 'weight', 'illDuration', 'allergy',
]


class DataSource(ABC):
    """数据源抽象接口：所有数据源对 CrawlerService 暴露统一契约。"""

    @abstractmethod
    def fetch(self, disease):
        """读取指定疾病类型的病例行，返回 list[tuple]，元素顺序与 CSV_FIELDS 一致。"""


class CsvDataSource(DataSource):
    """本地 CSV 文件数据源：按疾病类型过滤并返回病例行。"""

    def __init__(self, path):
        self.path = Path(path)

    def fetch(self, disease):
        rows = []
        try:
            with open(self.path, 'r', encoding='utf-8-sig', newline='') as f:
                for item in csv.DictReader(f):
                    if (item.get('type') or '').strip() != disease:
                        continue
                    rows.append(tuple((item.get(field) or '').strip() or None for field in CSV_FIELDS))
        except Exception as e:
            print('[DataSource] 读取数据源 {} 失败：{}'.format(self.path, e))
        return rows

    def __repr__(self):
        return 'CsvDataSource({})'.format(self.path)


class DataSourceFactory:
    """数据源工厂：维护「类型名 -> 数据源类」注册表，按类型创建实例。

    【工厂模式的核心】create() 把 if/else 判断收敛到一处；
    调用方只传类型字符串，不直接 new 具体类，从而与具体实现解耦。
    """

    _registry = {
        'csv': CsvDataSource,
    }

    @classmethod
    def register(cls, kind, source_cls):
        """注册新的数据源类型（扩展点：例如 register('web', WebCrawlerDataSource)）。"""
        if not issubclass(source_cls, DataSource):
            raise TypeError('数据源类必须继承 DataSource')
        cls._registry[kind] = source_cls

    @classmethod
    def create(cls, kind, **kwargs):
        """按类型创建数据源实例。未知类型立即报错，便于尽早发现配置错误。"""
        if kind not in cls._registry:
            raise ValueError('未知数据源类型: {}，已注册: {}'.format(kind, list(cls._registry)))
        return cls._registry[kind](**kwargs)

    @classmethod
    def discover_default_sources(cls):
        """扫描默认数据位置（temp2.csv 与 spiders/data_source/*.csv），
        通过工厂批量创建 CsvDataSource 实例。"""
        sources = []
        paths = []
        csv_default = BASE_DIR / 'temp2.csv'
        if csv_default.exists():
            paths.append(csv_default)
        paths.extend(Path(p) for p in glob.glob(str(DATA_SOURCE_DIR / '*.csv')))
        for path in paths:
            # 注意：此处不直接写 CsvDataSource(path)，而是走工厂创建，
            # 未来默认数据源换成其他类型时只需改这一行
            sources.append(cls.create('csv', path=path))
        return sources
