"""病例数据的统计查询。

app.py 的大屏接口 /getHomeData 依赖这里的 7 个函数，返回结构说明见各函数注释。
所有统计都实时查询 MySQL 的 cases 表，保证与数据管理页面看到的数据一致。
"""

from collections import Counter, defaultdict

import jieba

from utils.getPublicData import getSqlData, safe_float, safe_int

# cases 表业务字段顺序，与 app.py 中的 TEMPLATE_COLUMNS 保持一致（不含自增 id）
CASE_COLUMNS = (
    'id, type, gender, age, `time`, content, docName, docHospital, '
    'department, detailUrl, height, weight, illDuration, allergy'
)

# 词云过滤用的停用词（医疗文本里高频但无信息量的词）
STOP_WORDS = {
    '什么', '怎么', '如何', '需要', '可以', '是否', '这个', '那个', '现在', '目前',
    '医生', '请问', '谢谢', '帮忙', '看一下', '看下', '还是', '一下', '有点', '比较',
    '情况', '问题', '建议', '治疗', '检查', '患者', '时候', '已经', '没有', '但是',
    '就是', '感觉', '最近', '一直', '因为', '所以', '而且', '如果', '这样', '那样',
}


def getAllCasesData():
    """返回全部病例，按 id 升序。

    每条记录是元组：(id, type, gender, age, time, content, docName, docHospital,
    department, detailUrl, height, weight, illDuration, allergy)
    """
    return getSqlData('SELECT {} FROM cases ORDER BY id ASC'.format(CASE_COLUMNS))


def getPieData():
    """疾病类型占比，用于首页饼图。

    返回：[{'name': 疾病类型, 'value': 病例数}, ...]，按病例数降序。
    """
    rows = getSqlData(
        'SELECT type, COUNT(*) AS num FROM cases '
        'WHERE type IS NOT NULL AND type <> "" '
        'GROUP BY type ORDER BY num DESC'
    )
    return [{'name': row[0], 'value': int(row[1])} for row in rows]


def getConfigOne():
    """首页柱状图 + 病情描述词云。

    返回 (configOne, wordData)：
      configOne = {'type': [疾病类型...], 'sum': [对应病例数...]}，降序取前 10
      wordData  = [{'name': 关键词, 'value': 出现次数}, ...]，降序取前 60
    """
    rows = getSqlData(
        'SELECT type, COUNT(*) AS num FROM cases '
        'WHERE type IS NOT NULL AND type <> "" '
        'GROUP BY type ORDER BY num DESC LIMIT 10'
    )
    config_one = {
        'type': [row[0] for row in rows],
        'sum': [int(row[1]) for row in rows],
    }

    contents = getSqlData('SELECT content FROM cases WHERE content IS NOT NULL')
    counter = Counter()
    for (content,) in contents:
        for word in jieba.cut(str(content)):
            word = word.strip()
            if len(word) < 2 or word in STOP_WORDS:
                continue
            if not word.isalpha():
                continue
            counter[word] += 1
    word_data = [{'name': word, 'value': num} for word, num in counter.most_common(60)]

    return config_one, word_data


def getFoundData():
    """关键指标：病例最多的疾病、最忙的科室/医院、年龄区间。

    返回 (maxNum, maxType, maxDep, maxHos, maxAge, minAge)：
      maxNum  — 单一疾病的最大病例数
      maxType — 该疾病名称
      maxDep  — 病例数最多的科室名称
      maxHos  — 病例数最多的医院名称
      maxAge  — 最大患者年龄（无数据时为 0）
      minAge  — 最小患者年龄（无数据时为 0）
    """
    top_type = getSqlData(
        'SELECT type, COUNT(*) AS num FROM cases '
        'WHERE type IS NOT NULL AND type <> "" '
        'GROUP BY type ORDER BY num DESC LIMIT 1'
    )
    max_type, max_num = (top_type[0][0], int(top_type[0][1])) if top_type else ('暂无数据', 0)

    top_dep = getSqlData(
        'SELECT department, COUNT(*) AS num FROM cases '
        'WHERE department IS NOT NULL AND department <> "" '
        'GROUP BY department ORDER BY num DESC LIMIT 1'
    )
    max_dep = top_dep[0][0] if top_dep else '暂无数据'

    top_hos = getSqlData(
        'SELECT docHospital, COUNT(*) AS num FROM cases '
        'WHERE docHospital IS NOT NULL AND docHospital <> "" '
        'GROUP BY docHospital ORDER BY num DESC LIMIT 1'
    )
    max_hos = top_hos[0][0] if top_hos else '暂无数据'

    ages = [safe_int(row[0], default=-1) for row in getSqlData('SELECT age FROM cases')]
    ages = [age for age in ages if age > 0]
    max_age = max(ages) if ages else 0
    min_age = min(ages) if ages else 0

    return max_num, max_type, max_dep, max_hos, max_age, min_age


def getGenderData():
    """男女病例对比，用于首页性别柱状图与环形图。

    返回 (boyList, girlList, ratioData)：
      boyList  = [[疾病类型, 男性病例数], ...]
      girlList = [[疾病类型, 女性病例数], ...]
      两个列表的疾病顺序一致（按总病例数降序）
      ratioData = [{'name': '男', 'value': 男性总数}, {'name': '女', 'value': 女性总数}]
    """
    rows = getSqlData(
        'SELECT type, gender, COUNT(*) AS num FROM cases '
        'WHERE type IS NOT NULL AND type <> "" '
        'GROUP BY type, gender'
    )

    matrix = defaultdict(dict)
    total = Counter()
    for disease, gender, num in rows:
        gender = '男' if str(gender).strip() == '男' else '女'
        matrix[disease][gender] = int(num)
        total[disease] += int(num)

    disease_order = [d for d, _ in total.most_common()]
    boy_list = [[d, matrix[d].get('男', 0)] for d in disease_order]
    girl_list = [[d, matrix[d].get('女', 0)] for d in disease_order]

    total_boy = sum(item[1] for item in boy_list)
    total_girl = sum(item[1] for item in girl_list)
    ratio_data = [
        {'name': '男', 'value': total_boy},
        {'name': '女', 'value': total_girl},
    ]
    return boy_list, girl_list, ratio_data


def getCircleData():
    """科室分布环形图，取病例数前 8 的科室。

    返回：[{'name': 科室, 'value': 病例数}, ...]
    """
    rows = getSqlData(
        'SELECT department, COUNT(*) AS num FROM cases '
        'WHERE department IS NOT NULL AND department <> "" '
        'GROUP BY department ORDER BY num DESC LIMIT 8'
    )
    return [{'name': row[0], 'value': int(row[1])} for row in rows]


def getBodyData():
    """身高体重折线图：按年龄段统计平均身高、平均体重。

    返回 (xData, y1Data, y2Data)：
      xData  — 年龄段标签
      y1Data — 各年龄段平均身高（cm）
      y2Data — 各年龄段平均体重（kg）
    """
    groups = [
        ('0-18岁', 0, 18),
        ('19-30岁', 19, 30),
        ('31-45岁', 31, 45),
        ('46-60岁', 46, 60),
        ('60岁以上', 61, 200),
    ]

    rows = getSqlData('SELECT age, height, weight FROM cases')
    heights = defaultdict(list)
    weights = defaultdict(list)

    for age, height, weight in rows:
        age = safe_int(age, default=-1)
        if age < 0:
            continue
        for label, low, high in groups:
            if low <= age <= high:
                h = safe_float(height, default=0)
                w = safe_float(weight, default=0)
                if h > 0:
                    heights[label].append(h)
                if w > 0:
                    weights[label].append(w)
                break

    x_data = [label for label, _, _ in groups]
    y1_data = [
        round(sum(heights[label]) / len(heights[label]), 1) if heights[label] else 0
        for label in x_data
    ]
    y2_data = [
        round(sum(weights[label]) / len(weights[label]), 1) if weights[label] else 0
        for label in x_data
    ]
    return x_data, y1_data, y2_data
