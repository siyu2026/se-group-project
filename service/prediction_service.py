"""疾病预测服务（策略模式 + 单例模式），由 machine/tree.py 迁入并重构。

【设计模式说明 —— 策略模式（Strategy）】
定义一族算法（关键词规则打分 / TF-IDF 相似病例加权），
分别封装成实现同一 PredictionStrategy 接口的类，
运行期通过 PredictionService.set_strategy('keyword' | 'tfidf') 切换，
切换过程对调用方（app.py 路由）完全透明。

解决的问题：原 MedicalAI 把算法写死在类内部，换算法必须改源码；
现在新增算法只需：
    1. 继承 PredictionStrategy 实现 load/predict；
    2. 在 PredictionService._build_strategies 注册一行。
符合开闭原则：对扩展开放，对修改关闭。

【设计模式说明 —— 单例模式（Singleton）】
PredictionService 使用 SingletonMeta：TF-IDF 索引构建昂贵（全表分词 + 向量化），
全局只加载一次；定时采集入库新病例后调用 reload() 原地重建索引。

对外接口与原 MedicalAI 完全一致：
    prediction_service.load_model()
    prediction_service.predict(text) -> {'prediction': {...}, 'similar_cases': [...]}
"""

import threading
from abc import ABC, abstractmethod

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from service.base import SingletonMeta
from utils.getPublicData import getSqlData, safe_int

# ---------------------------------------------------------------- 常量配置
# 各疾病的典型症状关键词，用于症状文本的规则打分
DISEASE_KEYWORDS = {
    '高血压': ['血压', '高压', '低压', '降压', '血压高', '头晕', '头痛', '降压药'],
    '感冒': ['感冒', '咳嗽', '发热', '发烧', '流鼻涕', '鼻塞', '咽痛', '打喷嚏', '嗓子'],
    '腰椎间盘突出': ['腰', '腰椎', '间盘', '腰痛', '腿麻', '坐骨神经', '下肢', '弯腰'],
    '骨折': ['骨折', '摔', '外伤', '骨', '石膏', '车祸', '扭伤', '不能动'],
    '颈椎病': ['颈椎', '脖子', '颈', '肩', '手麻', '僵硬', '低头', '脖子疼'],
    '胃炎': ['胃', '胃炎', '胃痛', '反酸', '嗳气', '腹胀', '恶心', '胃镜', '烧心'],
    '抑郁症': ['抑郁', '情绪', '焦虑', '失眠', '心情', '兴趣', '烦躁', '睡不着', '压力'],
}

DISEASE_ADVICE = {
    '高血压': '建议规律监测血压并记录，低盐饮食、控制体重、戒烟限酒；降压药需遵医嘱长期服用，不要自行停药或加减量。',
    '感冒': '多为病毒感染，注意休息、多饮水，可对症使用退热止咳药物；若持续高热超过 3 天或出现呼吸困难，请及时就医。',
    '腰椎间盘突出': '急性期建议卧床休息、避免弯腰搬重物，可配合牵引、理疗与康复训练；若出现下肢无力或大小便异常需尽快就诊。',
    '骨折': '请尽快到骨科就诊拍片明确骨折类型，遵医嘱固定或手术；伤后早期冷敷、抬高患肢，避免自行揉按复位。',
    '颈椎病': '避免长时间低头，注意颈部保暖，可做颈椎操与热敷；若出现上肢麻木无力、行走不稳，建议做颈椎 MRI 检查。',
    '胃炎': '规律三餐、少食辛辣刺激与生冷食物，戒烟酒；建议完善幽门螺杆菌检测与胃镜检查，按医嘱使用抑酸与胃黏膜保护药。',
    '抑郁症': '情绪症状持续两周以上建议到精神心理科评估；保持规律作息与适度运动，家人陪伴与专业心理治疗同样重要，必要时遵医嘱用药。',
}

DEFAULT_ADVICE = '以上判断基于既往相似病例，仅供参考，不能替代医生面诊；建议携带相关检查报告到对应科室进一步确诊。'

SIMILAR_CASE_LIMIT = 5


def _tokenize(text):
    """jieba 分词并以空格连接，供 TF-IDF 向量化使用。"""
    return ' '.join(word for word in jieba.cut(str(text)) if word.strip())


def _build_result(disease, confidence, similar_cases):
    """组装统一返回结构（保持与原 MedicalAI.predict 一致）。"""
    advice = DISEASE_ADVICE.get(disease, '')
    advice = advice + (' ' + DEFAULT_ADVICE if advice else DEFAULT_ADVICE)
    return {
        'prediction': {
            'disease': disease,
            'confidence': confidence,
            'advice': advice,
        },
        'similar_cases': similar_cases,
    }


# ---------------------------------------------------------------- 策略接口
class PredictionStrategy(ABC):
    """预测策略抽象接口：所有算法对 PredictionService 暴露统一契约。"""

    #: 策略标识（供 set_strategy 使用）
    name = ''
    #: 策略说明（供接口查询展示）
    description = ''

    @abstractmethod
    def load(self, cases):
        """用病例数据初始化策略内部状态（如构建 TF-IDF 索引）。"""

    @abstractmethod
    def predict(self, text):
        """根据症状描述返回 {'prediction': {...}, 'similar_cases': [...]}。"""


class KeywordRuleStrategy(PredictionStrategy):
    """策略一：关键词规则打分。

    原理：统计症状文本中各疾病关键词的命中次数，命中最多者胜出；
    置信度随命中数线性增长并封顶 0.9。
    优点：零模型、可解释性强、速度快；缺点：无法处理近义表达。
    """

    name = 'keyword'
    description = '关键词规则打分（轻量、可解释，适合无训练数据场景）'

    def __init__(self):
        self._cases = []

    def load(self, cases):
        self._cases = cases

    def predict(self, text):
        scores = {}
        for disease, keywords in DISEASE_KEYWORDS.items():
            hit = sum(1 for keyword in keywords if keyword in text)
            if hit:
                scores[disease] = hit

        if not scores:
            return _build_result('未知', 0.0, [])

        disease = max(scores, key=scores.get)
        confidence = round(min(0.9, 0.45 + 0.1 * scores[disease]), 4)
        # 相似病例：取同疾病的前 N 条作为参考（本策略不计算相似度）
        similar_cases = [
            dict(case, similarity=None)
            for case in self._cases if case['type'] == disease
        ][:SIMILAR_CASE_LIMIT]
        return _build_result(disease, confidence, similar_cases)


class TfidfSimilarityStrategy(PredictionStrategy):
    """策略二：关键词规则 + TF-IDF 相似病例加权（原 MedicalAI 算法）。

    原理：
        1. 全量病例经 jieba 分词后构建 TF-IDF 向量索引；
        2. 输入症状同样向量化，与索引做余弦相似度，取 Top N 相似病例；
        3. 关键词命中给出规则置信度，再与最相似病例的相似度加权融合，
           避免纯规则过于武断、纯相似度缺乏先验。
    优点：能捕捉近义表达并给出相似病例佐证；缺点：依赖病例库规模。
    """

    name = 'tfidf'
    description = 'TF-IDF 相似病例 + 关键词加权（默认，兼顾准确率与可解释性）'

    def __init__(self):
        self._cases = []
        self._vectorizer = None
        self._matrix = None

    def load(self, cases):
        self._cases = cases
        corpus = [_tokenize('{} {}'.format(case['type'], case['content'])) for case in cases]
        self._vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self._matrix = self._vectorizer.fit_transform(corpus)

    def predict(self, text):
        vector = self._vectorizer.transform([_tokenize(text)])
        similarities = cosine_similarity(vector, self._matrix)[0]

        # 相似病例 Top N
        top_indexes = similarities.argsort()[::-1][:SIMILAR_CASE_LIMIT]
        similar_cases = [
            dict(self._cases[idx], similarity=round(float(similarities[idx]), 4))
            for idx in top_indexes
        ]

        # 关键词规则打分
        scores = {}
        for disease, keywords in DISEASE_KEYWORDS.items():
            hit = sum(1 for keyword in keywords if keyword in text)
            if hit:
                scores[disease] = hit

        if scores:
            disease = max(scores, key=scores.get)
            rule_conf = min(0.9, 0.45 + 0.1 * scores[disease])
            # 与最相似病例的相似度做加权，避免过于武断
            top_sim = max(
                (case['similarity'] for case in similar_cases if case['type'] == disease),
                default=0.0,
            )
            confidence = round(min(0.97, 0.6 * rule_conf + 0.4 * top_sim + 0.1), 4)
        else:
            # 未命中关键词时退化为相似病例投票
            disease = similar_cases[0]['type'] if similar_cases else '未知'
            confidence = round(float(similar_cases[0]['similarity']) if similar_cases else 0.0, 4)

        return _build_result(disease, confidence, similar_cases)


# ---------------------------------------------------------------- 预测服务
class PredictionService(metaclass=SingletonMeta):
    """预测服务单例：持有策略注册表与当前策略，委托具体算法完成预测。

    【策略模式的「上下文」角色】调用方只与本类交互，
    通过 set_strategy() 在运行期更换算法，无需感知具体策略类。
    """

    def __init__(self):
        self._strategies = self._build_strategies()
        self._strategy = self._strategies['tfidf']   # 默认策略
        self._ready = False
        self._lock = threading.Lock()

    @staticmethod
    def _build_strategies():
        """策略注册表：新增算法时在此注册一行即可。"""
        strategies = [TfidfSimilarityStrategy(), KeywordRuleStrategy()]
        return {s.name: s for s in strategies}

    # ------------------------------------------------------------ 策略管理
    def available_strategies(self):
        """返回全部可切换策略的标识与说明。"""
        return [
            {'name': s.name, 'description': s.description, 'active': s is self._strategy}
            for s in self._strategies.values()
        ]

    @property
    def current_strategy_name(self):
        return self._strategy.name

    def set_strategy(self, name):
        """运行期切换预测算法。切换后沿用已加载的病例数据，无需重建。"""
        if name not in self._strategies:
            raise ValueError('未知策略: {}，可选: {}'.format(name, list(self._strategies)))
        with self._lock:
            self._strategy = self._strategies[name]
        print('[Prediction] 已切换预测策略 -> {}'.format(name))

    # ------------------------------------------------------------ 数据加载
    def load_model(self):
        """从 cases 表加载病例并初始化所有策略（保持与原 MedicalAI 相同的接口名）。"""
        rows = getSqlData(
            'SELECT id, type, gender, age, content, docHospital, department '
            'FROM cases WHERE content IS NOT NULL AND content <> ""'
        )
        if not rows:
            raise FileNotFoundError('cases 表中暂无病例数据，请先导入 medicalinfo.sql')

        cases = [
            {
                'id': row[0],
                'type': row[1] or '未知',
                'gender': row[2] or '',
                'age': safe_int(row[3], default=0),
                'content': row[4] or '',
                'docHospital': row[5] or '',
                'department': row[6] or '',
            }
            for row in rows
        ]

        with self._lock:
            for strategy in self._strategies.values():
                strategy.load(cases)
            self._ready = True
        print('[Prediction] 已加载 {} 条病例，全部策略就绪'.format(len(cases)))

    def reload(self):
        """采集入库新病例后重建索引（供定时任务采集完成的回调调用）。"""
        try:
            self.load_model()
            print('[Prediction] 新数据入库，预测索引已重建')
        except Exception as e:
            print('[Prediction] 重建索引失败：{}'.format(e))

    # ------------------------------------------------------------ 预测入口
    def predict(self, text):
        """委托当前策略完成预测（接口与原 MedicalAI.predict 完全一致）。"""
        text = (text or '').strip()
        if not text:
            raise ValueError('症状描述不能为空')
        if not self._ready:
            self.load_model()
        return self._strategy.predict(text)


# 模块级单例实例
prediction_service = PredictionService()


if __name__ == '__main__':
    # 便于单独调试：python -m service.prediction_service
    prediction_service.load_model()
    result = prediction_service.predict('最近血压有点高，早上起来头晕，需要调整降压药吗')
    print('策略:', prediction_service.current_strategy_name)
    print(result['prediction'])
    prediction_service.set_strategy('keyword')
    print('切换后:', prediction_service.predict('最近血压有点高，头晕')['prediction'])
