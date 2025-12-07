# Feature-Selection-HW

此專案為碩一上之資料探勘課程之作業程式碼展示，目的在於實作資料前處理 (Data Preprocessing) 中的特徵選擇 (Feature Selection) 方法，用於評估特徵並搜尋特徵子集。

# 目標

* 從 UCI 機器學習資料庫  (UCI Machine Learning Repository) 下載 Breast Cancer Dataset。
* 設計程式以計算各個特徵與類別之間的 **對稱不確定性 (Symmetric Uncertainty, SU)**。
* 以 Forward Selection 與 Backward Selection 兩種策略搜尋最佳特徵子集。
* 評估各方法所得特徵子集的 **適合度 (Goodness)**。

# 架構簡述

資料讀取 → 計算 SU/Entropy → Goodness 評估 → Forward/Backward Selection → 輸出最佳特徵子集

主程式會根據作業要求執行：
* Forward Selection: 
  * 從空集合開始
  * 每一輪挑選能最大化 Goodness 的特徵加入子集
  * 持續到加入新特徵不再提升 Goodness

* Backward Elimination:
  * 從完整特徵集合開始
  * 每一輪移除對 Goodness 貢獻最小的特徵
  * 持續到移除特徵不再改善 Goodness

# 衡量指標

## 對稱不確定性 (Symmetric Uncertainty, SU)

$$SU(X, Y) = \frac{2 \times \big(H(X) + H(Y) - H(X, Y)\big)}{H(X) + H(Y)}$$
其中：
* $H(X)$：屬性 X 的熵
* $H(Y)$：屬性 Y 的熵
* $H(X,Y)$：屬性 X 與 Y 的聯合熵

透過 entropy 與 joint entropy 計算互資訊（Mutual Information），再透過正規化使 SU 的值介於 0 到 1，用來評估兩個屬性之間的關聯程度。
* `SU = 0`: 完全獨立
* `SU = 1`: 完全相互依賴

## 適合度 (Goodness)

$$Goodness(S) = \frac{\sum_{x_i \in S} SU(x_i, C)}{\sqrt{\sum_{x_i \in S}\sum_{x_j \in S} SU(x_i, x_j)}}$$
其中：
* $S$：特徵子集（feature subset）
* $C$：類別屬性（Class）

goodness 同時考量每個特徵與 Class 的相關程度，以及特徵間彼此的冗餘程度，以一個正規化後的適合度量值衡量特徵子集的品質。
* 分子：衡量特徵集合和 Class 的相關性
* 分母：衡量特徵彼此之間的冗餘程度（越大代表特徵間越相似）

Goodness 越高，代表該特徵子集與 Class 越相關，且每個特徵之間的關聯性低。


# 執行環境與方式

Python 3.8 或以上環境，
直接執行 .py 或 .ipynb 檔案即可。 

# 未來可增加部分

* 增加可視化圖表
* 容器化