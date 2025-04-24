# 📊 Facebook 贴文互动分析：基于图文内容与发布时间的实证研究

> 本项目结合 A/B 测试、回归模型、倾向得分匹配（PSM）与 XGBoost 特征解释，探索在 Facebook 平台上，**什么因素决定了一条贴文能获得更多评论**。

---

## 📌 项目背景

本项目基于 [UCI Facebook Comment Volume 数据集](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset)，模拟品牌/平台在实际运营中可能遇到的内容投放问题：

- 图文内容是否比文字更容易获得用户评论？
- 发布时间是否与内容类型存在联动效应？
- 在控制页面热度与历史表现后，这些结论是否依然成立？

---

## 🔬 分析方法概览

| 步骤 | 方法 | 目的 |
|------|------|------|
| 1️⃣ | EDA + 可视化 | 初步观察评论分布与内容类型差异 |
| 2️⃣ | A/B 测试 + CUPED | 测量图文与文字内容的初始效应，并进行方差调整 |
| 3️⃣ | OLS 回归 + 交互项 | 检验图文效果是否受发布时间影响 |
| 4️⃣ | 混合效应模型（Mixed Effect） | 控制内容来源（source_file）的随机效应，增强稳健性 |
| 5️⃣ | PSM 倾向得分匹配 | 样本匹配，缓解选择偏差 |
| 6️⃣ | XGBoost + 特征重要性 | 挖掘影响评论数的关键变量，提升可解释性

---

## 📈 核心发现总结

- **图文内容的评论效果依赖于投放时间段**：早上表现差，晚上效果最好。
- **图文的平均效应在控制协变量后变得非常微弱（Cohen's d < 0.02）**，说明其本身不是决定评论数的关键。
- **是否推广（is_paid）对评论量影响极为显著**，但需结合时段进行预算优化。
- **XGBoost 显示发帖时间（Post_Hour）为影响评论的最重要特征**。

---

## 💡 商业洞察与策略建议

1. **图文内容的有效性依赖于投放时段**：傍晚与夜间投放更具评论潜力；
2. **非活跃时段需推广辅助**：中午/清晨时段建议搭配广告预算进行曝光；
3. **内容排程建议系统化建设**：结合模型输出制定智能推送节奏；
4. **该方法论可迁移到微博、Instagram 等平台**，作为内容运营的科学决策工具。


# 📊 Facebook Engagement A/B Test: An Empirical Study on Post Format & Timing

> A causal analysis of what drives user interaction on Facebook posts — combining A/B testing, regression models, PSM matching and XGBoost-based feature ranking.

---

## 📌 Project Background

This project is based on the [UCI Facebook Comment Volume Dataset](https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset), and simulates a real-world scenario where a content operations team wants to evaluate:

- Does **photo-based content** lead to more engagement than text?
- Does **posting time** interact with **post format**?
- Is the effect robust after adjusting for baseline user/page characteristics?

---

## 🔬 Methodology

We adopt a **multi-method empirical strategy**:

| Step | Method | Purpose |
|------|--------|---------|
| 1️⃣ | EDA + Visualization | Understand basic distribution of post types and comment volume |
| 2️⃣ | A/B Test + CUPED | Estimate raw and variance-adjusted effect of photo vs text |
| 3️⃣ | OLS Regression + Interactions | Test for heterogeneous effects over time slots |
| 4️⃣ | Mixed Effects Model | Control for source-file level random effects |
| 5️⃣ | Propensity Score Matching (PSM) | Robustness check for treatment-control balancing |
| 6️⃣ | XGBoost + Feature Importance | Identify key drivers of high-comment posts |

---

## 📈 Key Findings

- **Posting time significantly moderates the effect of photo-based content**  
  → Visual posts perform worse in early hours, better in the evening.

- **The treatment effect of using photo is small in raw form**, but after adjustment, **effect is marginal (Cohen's d < 0.02)**.

- **Promotion significantly boosts engagement**, but **returns diminish in peak time slots**.

- **Post_Hour is the most important driver** in predicting comment volume, per XGBoost ranking.

---

## 💡 Business Takeaways

1. **Post format alone doesn’t guarantee more engagement** — context like time and promotion matters.
2. To improve ROI, **allocate paid promotion for photo content in off-peak hours**.
3. Use model predictions to build **automated content push scheduling systems**.
4. This methodology can be applied to other platforms (Instagram, Twitter) for content strategy optimization.
