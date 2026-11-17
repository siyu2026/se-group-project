"""单例模式的统一实现：SingletonMeta 元类。

【设计模式说明 —— 单例模式（Singleton）】
目的：保证一个类在进程内只有一个实例，并提供全局访问点。
本项目中的适用场景：
    - PredictionService：TF-IDF 索引构建昂贵，必须只加载一次并全局共享；
    - CrawlerService：CSV 数据源清单与去重状态需要全局唯一；
    - SchedulerService：APScheduler 只能启动一份，否则任务会重复触发。

实现原理：元类拦截「类调用」——当代码写 SomeSingleton() 时，
实际执行的是元类的 __call__。这里用「双重检查锁」保证多线程下
也只创建一个实例，且只有首次创建需要加锁，后续访问零开销。
"""

import threading


class SingletonMeta(type):
    """线程安全的单例元类。把它作为 metaclass 的类即成为单例。"""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # 第一次检查：已创建则直接返回，避免每次访问都加锁
        if cls not in cls._instances:
            with cls._lock:
                # 第二次检查：防止多个线程同时穿过第一次检查后重复创建
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
