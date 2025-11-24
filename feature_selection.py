import math  # 計算對數用

# 0~9代表屬性，其中Attribute[0] = Class 是我們要比較的對象
Attribute = [
    "Class",
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
]

# 經整理過後的資料
patient_list = []

# 讀取檔案並整理格式
with open("breast-cancer.txt", "r") as file:
    for line in file:
        data = line.strip().split(",")
        (
            Class,
            age,
            menopause,
            tumor_size,
            inv_nodes,
            node_caps,
            deg_malig,
            breast,
            breast_quad,
            irradiat,
        ) = data
        next_patient_data = {
            "Class": Class,
            "age": age,
            "menopause": menopause,
            "tumor_size": tumor_size,
            "inv_nodes": inv_nodes,
            "node_caps": node_caps,
            "deg_malig": deg_malig,
            "breast": breast,
            "breast_quad": breast_quad,
            "irradiat": irradiat,
        }
        patient_list.append(next_patient_data)


def compute_prob(data):
    """
    計算各個屬性中，每一種可能值的出現機率
    用於計算 entropy

    參數:
        data: List[Dict[str, str]]
            已格式化的病人資料

    返回:
        prob: List[Dict[str, float]]
            列表內存放同屬性數量的字典，
            紀錄每個屬性中可能值的機率分布
    """
    num_of_attr = len(Attribute)

    # 存放各個屬性可能值的機率
    prob = []
    for _ in range(num_of_attr):
        prob.append({})

    # 計算每一種可能值的出現次數
    for d in data:
        for attr_idx in range(num_of_attr):
            if d.get(Attribute[attr_idx]) in prob[attr_idx].keys():
                prob[attr_idx][d.get(Attribute[attr_idx])] = (
                    prob[attr_idx][d.get(Attribute[attr_idx])] + 1
                )
            else:
                prob[attr_idx][d.get(Attribute[attr_idx])] = 1

    # 以總資料筆數求機率
    total = len(data)
    for attr_prob in prob:
        for key, value in attr_prob.items():
            attr_prob[key] = value / total

    return prob


def entropy(x, data):
    """
    計算屬性 x 的 entropy: H(x)
    用於計算 entropy 與 SU

    參數:
        x: str
            屬性名稱
        data: List[Dict[str, str]]
            已格式化的病人資料

    返回:
        float
            屬性 x 的entropy
    """
    # 找到屬性在 Attribute 中的索引
    attr_idx = Attribute.index(x)

    # 取得該屬性的機率分佈（dict: value -> prob）
    attr_prob = compute_prob(data)[attr_idx].values()

    # 計算 entropy
    return -sum(p * math.log2(p) for p in attr_prob)


def joint_entropy(x, y, data):
    """
    計算屬性 x, y 的 joint entropy: H(x, y)
    用於計算 SU

    參數:
        x, y: str
            屬性名稱
        data: List[Dict[str, str]]
            已格式化的病人資料

    返回:
        float
            屬性 x, y 的 joint entropy
    """
    # 計算所有 (x, y) 組合的出現次數
    pair_count = {}
    for d in data:
        pair = (d[x], d[y])
        if pair in pair_count:
            pair_count[pair] += 1
        else:
            pair_count[pair] = 1

    total = len(data)

    # 計算機率
    prob = [count / total for count in pair_count.values()]

    # 計算 joint entropy
    return -sum(p * math.log2(p) for p in prob)


# 計算symmetric_uncertainty
def su(x, y, data):
    """
    計算 symmetric uncertainty
    用於計算 Goodness

    參數:
        x, y: str
            屬性名稱
        data: List[Dict[str, str]]
            已格式化的病人資料

    返回:
        float
            屬性 x, y 的 joint entropy
    """
    hx = entropy(x, data)
    hy = entropy(y, data)
    hxy = joint_entropy(x, y, data)

    denominator = hx + hy
    numerator = denominator - hxy

    # 避免分母為 0
    if denominator == 0:
        return 0.0

    return 2 * numerator / denominator


def goodness(subset, data):
    """
    計算 Goodness

    參數:
        subset: List[str]
            當前特徵子集包含的屬性
        data: List[Dict[str, str]]
            已格式化的病人資料

    返回:
        float
            特徵子集的 Goodness
    """

    # 計算分子
    numerator = 0.0
    target = Attribute[0]  # Class

    for feature in subset:
        numerator += su(feature, target, data)

    # 計算分母
    if len(subset) == 1:
        denominator = 1.0
    else:
        denominator = 0.0
        for f1 in subset:
            for f2 in subset:
                denominator += su(f1, f2, data)

    # 避免分母為 0
    if denominator == 0:
        return 0.0

    return numerator / math.sqrt(denominator)


# Forward Selection

print("===== Forward Selection=====")

f_feature_subset = []
f_remaining_attributes = [
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
]
f_best_goodness = 0.0
f_iteration = 0

while True:
    best_goodness_in_cycle = 0.0
    f_iteration = f_iteration + 1

    # 每次在子集裡尋找一個特徵，使得將其加入子集後 Goodness 是最好
    for attr_idx in range(len(f_remaining_attributes)):
        new_feature_subset = f_feature_subset.copy()
        new_feature_subset.append(f_remaining_attributes[attr_idx])

        if goodness(new_feature_subset, patient_list) > best_goodness_in_cycle:
            feature_to_add = f_remaining_attributes[attr_idx]
            best_goodness_in_cycle = goodness(new_feature_subset, patient_list)

    # 如果加入特徵不能再使 Goodness 增加，則停止
    if best_goodness_in_cycle < f_best_goodness:
        break

    # 如果可以，則更新子集和 Goodness
    f_feature_subset.append(feature_to_add)
    f_best_goodness = best_goodness_in_cycle
    f_remaining_attributes.remove(feature_to_add)

    print(f"第{f_iteration}次循環:")
    print(f"Selected Features: {f_feature_subset}")
    print(f"Best Goodness: {f_best_goodness:}")

print("Forward Selection result:", f_feature_subset, ", Gn: ", f_best_goodness)


# Backward Selection

print("===== Backward Selection=====")


b_feature_subset = [
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
]
b_best_goodness = goodness(b_feature_subset, patient_list)
b_iteration = 0


while True:
    best_goodness_in_cycle = 0.0
    b_iteration = b_iteration + 1

    # 每次在子集裡尋找一個特徵，使得將其剔除後 Goodness 是最好
    for attr_idx in range(len(b_feature_subset)):
        new_feature_subset = b_feature_subset.copy()
        new_feature_subset.remove(b_feature_subset[attr_idx])

        # 檢查剔除該元素可否使 Goodness 增加，可則將欲元素和新的 Goodness 紀錄
        if goodness(new_feature_subset, patient_list) > best_goodness_in_cycle:
            feature_to_remove = b_feature_subset[attr_idx]
            best_goodness_in_cycle = goodness(new_feature_subset, patient_list)

    # 如果剔除屬性不能再使 Goodness 增加，則停止
    if best_goodness_in_cycle < b_best_goodness:
        break

    # 若可以，則更新子集與 Gn
    b_feature_subset.remove(feature_to_remove)
    b_best_goodness = best_goodness_in_cycle

    print(f"第{b_iteration}次循環:")
    print(f"Selected Features: {b_feature_subset}")
    print(f"Best Goodness: {b_best_goodness:}")

print("Backward Selection result:", b_feature_subset, ",Gn: ", b_best_goodness)
