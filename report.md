# 每日作业报告

## 1. 本日问题

- 里程碑：day-04
- 学生或小组：为ai发电
- 使用者：需要从历史档案中找证据、且不能凭空回答的新闻编辑研究助理
- 真实输入：Kaggle BBC News Archive `tfidf_dataset.csv`（2,225 篇真实文章，business/entertainment/politics/sport/tech 五类）
- 需要的输出：对自然语言问题按余弦相似度返回 top-k 文章与文档编号；支持时引用、无支持时拒绝
- 与使用者最相关的错误：档案无证据仍然回答（编造），或引用与结论不一致
- 本日产品边界：检索基于 TF-IDF 词频表示，不保证理解语义；模型路线需教师批准

## 2. 真实数据或真实课程输入

- 所有者/发布者：Kaggle 用户 dimasmunoz（源自 BBC 文章语料）
- 标题：BBC News Archive (cleaned)
- 原始 URL：https://www.kaggle.com/datasets/dimasmunoz/bbc-articles-cleaned
- 许可标签或使用许可：Kaggle 数据集许可，仅限本课程用途
- 下载/取得日期：2026-08-17
- 预期文件与结构：`data/raw/tfidf_dataset.csv`，2,225 行，含 text 与 category 列
- 检查命令：`python app.py --evaluate`（检索评估，加载时即校验 2,225 行与五类）
- 实际检查结果：三个评估问题全部 PASS：musician_visas、psp_launch、phone_virus；`retrieval_recall_at_4=3/3`
- 已知缺失、偏差或限制：TF-IDF 检索以词重叠为相似度依据，同义改写可能检索失败；档案年代固定

## 3. 可复现运行

```powershell
# 当前目录
ai-camp-2026-deploy\day-04-grounded-agent

# 安装
python -m pip install -r requirements.txt

# 检索评估（数据检查 + 三个固定问题）
python app.py --evaluate

# 测试
python -m unittest discover -s tests -v

# 档案内问题检索（Windows 控制台建议 -X utf8）
python -X utf8 app.py --question "When and at what price was Sony's PSP expected to launch in Europe?"

# 档案外问题（应拒绝回答）
python -X utf8 app.py --question "Who won the 2024 Olympic men's football tournament?"
```

关键预期输出：评估 `PASS` × 3、`retrieval_recall_at_4=3/3`；测试 `Ran 4 tests ... OK`。

## 4. 基线与候选

### 简单基线

- 方法：无模型检索——直接用 TF-IDF 向量 + 余弦相似度，这是信息检索最低限度的比较对象
- 为什么足够简单：不涉及任何外部模型或生成，纯统计表示
- 命令：`python app.py --evaluate`
- 结果：`retrieval_recall_at_4=3/3`，三个固定问题在 top-4 内命中期望文章

### 候选方法

- 学生完成的核心改动：`retriever.py` 的 `ArchiveIndex.search`（问题向量化 → 余弦相似度 → 降序取 top-k）
- 保持不变的条件：同一 2,225 篇文章、同一 TF-IDF 参数（1-2 元词、英文停用词）、同一三个评估问题与 top-4
- 命令：`python app.py --evaluate`
- 结果：musician_visas → article-0000（score 0.231）；psp_launch → 含 article-1837（0.192）、article-1946（0.184）；phone_virus → 含 article-2224（0.191）

| 项目 | 基线 | 候选 | 含义 |
| --- | ---: | ---: | --- |
| retrieval_recall@4 | 3/3（同一实现） | 3/3 | 固定问题可复现命中 |
| 档案外问题 | 相似度 0.10–0.13 | 相似度 0.10–0.13 | 旧文章仍可能被排序到前面，必须拒绝 |

## 5. 一个真实失败案例

- 样本位置/编号：问题 "Who won the 2024 Olympic men's football tournament?" 的检索结果
- 真实结果：档案（约 2004–2005 年 BBC 文章）不包含该信息
- 系统输出：返回了 4 篇 2004 年奥运相关文章，最高相似度 0.127（如 article-1281 关于 Kelly Holmes）
- 可以观察到什么：TF-IDF 因为"Olympic/football"等词重叠把旧文章排到前面，但这些文章不回答 2024 问题
- 说明的限制：相似度 ≠ 答案存在性；词重叠检索对时间敏感问题不可靠
- 不能证明什么：不能证明这 4 篇文章回答了 2024 问题，也不能证明"档案没有答案"——只能说明检索结果不支持该问题
- 下一项最小检查：在 `--answer` 路径加入 min-score 与年代边界判断，对相似度低于阈值的检索明确输出拒绝语句

## 6. 智能体与学生工作边界

- 智能体提出/生成/修改了什么：智能体实现了 `retriever.py` 的 `ArchiveIndex.search` 一个 TODO，并生成报告草稿
- 学生怎样核对文件、来源、输出、测试和 diff：运行评估和测试核对 3/3 与 4 个测试 OK；打开 `test_retrieval.py` 确认测试未改；用 `git diff` 确认只改 `retriever.py`
- 学生修改或拒绝了什么建议：拒绝把相似度阈值下调到 0.05 让边缘问题"通过"；拒绝在没有教师批准密钥时调用 DeepSeek 路线
- 每名成员能独立解释的代码或证据：`cosine_similarity(query_vector, self.matrix)` 的形状与排序、`citation_ids` 的正则、`validate_citations` 的引用校验逻辑

## 7. 结论与限制

1. TF-IDF + 余弦检索在三个固定问题上 top-4 全部命中（3/3），检索实现可复现。2. PSP 问题同时命中 article-1837 和 article-1946，说明多文档召回有效。3. 对 2024 年问题，检索返回 2004 年旧文章且相似度低（≤0.13），系统无法通过检索本身判断答案不存在。4. 数据限制：TF-IDF 是词重叠模型，同义改写、专有名词变体会降低召回。5. 方法限制：无模型路线只排序不回答，回答与拒绝的质量依赖人工或经批准的模型判断。6. 使用边界：文章文字只当资料、不当指令；不把检索到的文字当作事实。7. 不能用于真实决策：该档案固定且不更新，不能支撑 2005 年以后的新闻结论。

## 8. 提交复核

- [x] README 从新环境可以开始运行
- [x] 数据检查、测试和主程序重新运行
- [x] 报告数字与保存输出一致
- [x] `presentation.pptx` 在 3 分钟内讲完
- [x] `submission.json` 路径正确
- [x] 无密钥、大数据、私人信息、虚拟环境或缓存
- [ ] GitHub 网页复查并邮件发送 URL（由学生本人完成）
