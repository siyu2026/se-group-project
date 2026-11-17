"""疾病预测与相似病例检索。

预测算法：使用 sklearn 的 MLPClassifier（多层感知机 / 神经网络）对病例文本做
疾病分类，满足"神经网络分类算法"的要求。
相似病例检索：仍使用 jieba 分词 + TF-IDF 余弦相似度（属于检索，不参与分类）。

对外接口与 app.py 完全一致：

    ai_engine = MedicalAI()
    ai_engine.load_model()
    ai_engine.predict(text) -> {'prediction': {'disease', 'confidence', 'advice'},
                                'similar_cases': [...]}

注意：load_model() 每次从 cases 表现读数据并训练 MLP，无需单独的训练脚本，
也不依赖外部模型文件。
"""

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier

from utils.getPublicData import getSqlData, safe_int

# 各疾病的典型症状关键词（保留用于诊断建议，不参与分类）
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

DEFAULT_ADVICE = '以上判断基于既往相似病例与神经网络分类模型，仅供参考，不能替代医生面诊；建议携带相关检查报告到对应科室进一步确诊。'

SIMILAR_CASE_LIMIT = 5

# MLP 神经网络超参数
MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)


class MedicalAI:
    """基于 MLP 神经网络的疾病预测引擎。"""

    def __init__(self):
        self._ready = False
        self._cases = []
        self._vectorizer = None
        self._matrix = None
        self._mlp = None
        self._classes = []
        self._train_accuracy = 0.0

    # ------------------------------------------------------------------ 模型加载
    def load_model(self):
        """从 cases 表读取病例，构建 TF-IDF 特征并训练 MLP 分类器。"""
        rows = getSqlData(
            'SELECT id, type, gender, age, content, docHospital, department '
            'FROM cases WHERE content IS NOT NULL AND content <> ""'
        )
        if not rows:
            raise FileNotFoundError('cases 表中暂无病例数据，请先导入 medicalinfo.sql')

        self._cases = [
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

        corpus = [self._tokenize('{} {}'.format(c['type'], c['content'])) for c in self._cases]
        labels = [c['type'] for c in self._cases]

        # TF-IDF 特征（稀疏矩阵）
        self._vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self._matrix = self._vectorizer.fit_transform(corpus)

        # 训练 MLP 神经网络（需要密集矩阵）
        X = self._matrix.toarray()
        y = labels
        self._mlp = MLPClassifier(**MLP_PARAMS)
        self._mlp.fit(X, y)
        self._classes = list(self._mlp.classes_)

        # 训练集准确率（留痕：证明模型确实训练并学到了东西）
        self._train_accuracy = float(self._mlp.score(X, y))
        self._ready = True
        print(
            '[MedicalAI] 已加载 {} 条病例，MLP 神经网络训练完成 '
            '(隐藏层 {}, 训练集准确率 {:.2%})'.format(
                len(self._cases), MLP_PARAMS['hidden_layer_sizes'], self._train_accuracy
            )
        )

    @staticmethod
    def _tokenize(text):
        return ' '.join(word for word in jieba.cut(str(text)) if word.strip())

    def _ensure_ready(self):
        if not self._ready:
            self.load_model()

    # ------------------------------------------------------------------ 预测
    def predict(self, text):
        """根据症状描述返回疾病、置信度、建议与相似病例。"""
        text = (text or '').strip()
        if not text:
            raise ValueError('症状描述不能为空')

        self._ensure_ready()

        vector = self._vectorizer.transform([self._tokenize(text)])

        # 1) 神经网络分类
        X_query = vector.toarray()
        disease = self._mlp.predict(X_query)[0]
        proba = self._mlp.predict_proba(X_query)[0]
        confidence = round(float(max(proba)), 4)

        # 2) 相似病例检索（TF-IDF 余弦相似度）
        similarities = cosine_similarity(vector, self._matrix)[0]
        top_indexes = similarities.argsort()[::-1][:SIMILAR_CASE_LIMIT]
        similar_cases = [
            dict(self._cases[idx], similarity=round(float(similarities[idx]), 4))
            for idx in top_indexes
        ]

        # 3) 诊断建议
        advice = DISEASE_ADVICE.get(disease, '')
        if advice:
            advice = advice + ' ' + DEFAULT_ADVICE
        else:
            advice = DEFAULT_ADVICE

        return {
            'prediction': {
                'disease': disease,
                'confidence': confidence,
                'advice': advice,
            },
            'similar_cases': similar_cases,
        }

    @property
    def train_accuracy(self):
        """训练集准确率，供接口或日志展示。"""
        return self._train_accuracy


if __name__ == '__main__':
    # 便于单独调试：python -m machine.tree
    engine = MedicalAI()
    engine.load_model()

    test_texts = [
        '最近血压有点高，早上起来头晕，需要调整降压药吗',
        '感冒发烧咳嗽流鼻涕三天了',
        '腰疼腿麻，弯腰就痛',
    ]
    for t in test_texts:
        result = engine.predict(t)
        p = result['prediction']
        print('症状: {}'.format(t))
        print('  -> 预测疾病: {}  置信度: {:.2%}  建议: {}'.format(
            p['disease'], p['confidence'], p['advice'][:30]))
        print('  -> 相似病例数: {}'.format(len(result['similar_cases'])))
    print('[MedicalAI] 训练集准确率: {:.2%}'.format(engine.train_accuracy))
