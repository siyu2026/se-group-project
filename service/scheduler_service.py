"""定时采集调度服务（单例模式 + 观察者模式），解决"无定时任务、非实时"问题。

【APScheduler 原理】
BackgroundScheduler 在 Flask 进程内启动一个守护线程，内部事件循环：
    1. 计算所有 job 的下次触发时间；
    2. 睡眠到最近的触发点；
    3. 唤醒后将到期 job 提交给内部线程池执行。
与 Linux cron 的区别：它是「进程内调度」，任务函数可直接访问
本进程的单例服务（CrawlerService）与 WebSocket 广播回调，无需跨进程通信。

关键参数：
    - max_instances=1：上一轮采集未结束时不并发重入，防止重复写入；
    - coalesce=True：服务停机期间错过的多次触发，重启后合并为一次补跑；
    - misfire_grace_time：允许任务"迟到"执行的宽限秒数。

【观察者模式（通知机制）】
本服务不直接依赖 Flask / SocketIO。采集完成后遍历 _listeners 回调列表
广播结果；由 app.py 注册一个「发 WebSocket」的 listener。
这样 service 层保持零 Web 依赖，可独立单元测试。

【单例模式】APScheduler 全局只能启动一份，故使用 SingletonMeta。
"""

import threading
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from service.base import SingletonMeta


class SchedulerService(metaclass=SingletonMeta):
    """封装 APScheduler，周期性执行采集任务并广播结果（单例 + 观察者）。"""

    def __init__(self):
        self._scheduler = None
        self._job = None
        self._listeners = []       # 观察者列表：采集完成后的回调 fn(result dict)
        self._lock = threading.Lock()
        self.last_run_at = None    # 最近一次触发时间
        self.last_results = []     # 最近一次各数据源采集结果

    # ------------------------------------------------------------- 观察者注册
    def add_listener(self, fn):
        """注册采集完成监听器。fn 接收一个 dict：{code, disease, delta, success, error}。"""
        self._listeners.append(fn)

    def _notify(self, result):
        """依次通知所有观察者；单个观察者异常不影响其他观察者。"""
        for fn in self._listeners:
            try:
                fn(result)
            except Exception as e:
                print('[Scheduler] 通知观察者失败：{}'.format(e))

    # ------------------------------------------------------------- 任务执行
    def run_once(self, crawl_executor):
        """执行一轮全量采集：对所有启用的数据源各跑一次。

        crawl_executor 由 app.py 注入，签名 fn(code) -> result dict，
        内部完成：状态机更新、调用 CrawlerService、版本快照记录。
        采用注入而非直接 import，是为了让本服务不依赖 app.py（避免循环依赖、
        保持 service 层可独立测试）。
        """
        with self._lock:
            self.last_run_at = datetime.now()
            results = []
            print('[Scheduler] 定时采集开始 @ {}'.format(self.last_run_at.strftime('%Y-%m-%d %H:%M:%S')))
            try:
                source_codes = self._enabled_source_codes()
            except Exception as e:
                result = {'code': None, 'disease': None, 'delta': 0,
                          'success': False, 'error': str(e)}
                results.append(result)
                self._notify(result)
                self.last_results = results
                print('[Scheduler] 定时采集无法开始：{}'.format(e))
                return results
            if not source_codes:
                result = {'code': None, 'disease': None, 'delta': 0,
                          'success': False, 'error': '没有启用的数据源'}
                results.append(result)
                self._notify(result)
            for code in source_codes:
                try:
                    result = crawl_executor(code, trigger='schedule')
                except Exception as e:
                    result = {'code': code, 'disease': code, 'delta': 0,
                              'success': False, 'error': str(e)}
                results.append(result)
                # 无论成功失败都广播，让前端实时看到状态变化
                self._notify(result)
            self.last_results = results
            total_delta = sum(r.get('delta', 0) for r in results)
            print('[Scheduler] 定时采集结束，共新增 {} 条'.format(total_delta))
            return results

    @staticmethod
    def _enabled_source_codes():
        """从 data_sources 表读取启用的数据源编码；表不可用时返回空列表。"""
        from utils.db_config import get_db_connection
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute('SELECT code FROM data_sources WHERE enabled=1')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            raise RuntimeError('读取启用数据源失败：{}'.format(e)) from e
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------- 调度控制
    def start(self, crawl_executor, interval_minutes=30, run_immediately=False):
        """启动后台调度器。

        interval_minutes: 采集周期（分钟）。
        run_immediately:  启动后立即先跑一轮（默认 False，等第一个周期）。
        """
        if self._scheduler is not None:
            print('[Scheduler] 调度器已在运行，忽略重复启动')
            return self

        self._crawl_executor = crawl_executor
        self._scheduler = BackgroundScheduler(daemon=True)
        self._job = self._scheduler.add_job(
            self.run_once,
            trigger='interval',
            minutes=interval_minutes,
            args=[crawl_executor],
            id='crawl_all_sources',
            max_instances=1,        # 防止上一轮未结束时并发重入
            coalesce=True,          # 错过的触发合并为一次补跑
            misfire_grace_time=300, # 允许迟到 5 分钟
        )
        self._scheduler.start()
        print('[Scheduler] 定时调度已启动：每 {} 分钟采集一次'.format(interval_minutes))

        if run_immediately:
            # 单独起线程执行首轮，避免阻塞 Flask 启动
            threading.Thread(target=self.run_once, args=[crawl_executor], daemon=True).start()
        return self

    def shutdown(self):
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            print('[Scheduler] 调度器已停止')

    @property
    def running(self):
        return self._scheduler is not None

    @property
    def next_run_time(self):
        """下次触发时间（供接口查询，前端展示用）。"""
        if self._job is not None and self._scheduler is not None:
            return self._job.next_run_time
        return None


# 模块级单例实例
scheduler_service = SchedulerService()
