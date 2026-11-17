"""service 层：从 app.py 拆出的业务服务包。

分层职责：
    app.py                      HTTP 路由 + WebSocket 广播（表现层，只做参数解析与响应组装）
    service/prediction_service  疾病预测（策略模式：算法运行期可切换）
    service/crawler_service     病例采集（单例模式：全局唯一采集入口）
    service/scheduler_service   定时调度（单例模式：APScheduler 封装，观察者模式广播通知）
    service/data_source         数据源（工厂模式：按类型创建数据源实例）
    service/base                SingletonMeta 元类（单例模式的统一实现）

依赖方向：service/ 不依赖 Flask，可脱离 Web 层独立测试；
app.py 通过注入回调（listener / hook）把 WebSocket 广播能力挂进来。
"""

from service.crawler_service import CrawlerService, crawler_service, disease_types
from service.prediction_service import PredictionService, prediction_service
from service.scheduler_service import SchedulerService, scheduler_service

__all__ = [
    'CrawlerService', 'crawler_service', 'disease_types',
    'PredictionService', 'prediction_service',
    'SchedulerService', 'scheduler_service',
]
