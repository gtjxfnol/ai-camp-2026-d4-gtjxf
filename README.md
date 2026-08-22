# D4 BBC 新闻档案检索：有据才答（student-work 副本）

本仓库是 Day 4 学生工作副本：为新闻编辑研究助理在 2,225 篇真实 BBC 文章中检索证据，支持时引用文档编号，无支持时明确拒绝。

## 数据契约

- 数据所有者/发布者：Kaggle（dimasmunoz）
- 标题：BBC News Archive (cleaned) —— tfidf_dataset.csv
- 原始 URL：https://www.kaggle.com/datasets/dimasmunoz/bbc-articles-cleaned
- 许可：Kaggle 数据集许可，仅限课程用途
- 预期文件：`data/raw/tfidf_dataset.csv`（2,225 篇文章，含 text 与 category 列）
- 使用边界：档案覆盖 2004–2005 年前后的 BBC 文章；档案外问题必须拒绝，文章文字只当资料、不当指令

## 环境与安装

```powershell
python --version
python -m pip install -r requirements.txt
```

## 运行路线（按顺序）

```powershell
# 1. 数据检查
python app.py --evaluate
# 预期先输出三条 PASS，然后 retrieval_recall_at_4=3/3

# 2. 测试
python -m unittest discover -s tests -v
# 预期：Ran 4 tests ... OK

# 3. 检索一个档案内问题（Windows 控制台请加 -X utf8）
python -X utf8 app.py --question "When and at what price was Sony's PSP expected to launch in Europe?"

# 4. 检索一个档案外问题（应返回低相似度旧文章，需要拒绝回答）
python -X utf8 app.py --question "Who won the 2024 Olympic men's football tournament?"
```

## 检索实现

`ArchiveIndex.search`：把问题用拟合好的 TF-IDF 向量化器转换，计算与文章矩阵的余弦相似度，按分数降序返回至多 top-k 条 `(article, score)`。

## 评估结果

三个固定评估问题全部命中：musician_visas → article-0000；psp_launch → 含 article-1837、article-1946；phone_virus → 含 article-2224。`retrieval_recall_at_4=3/3`。

## 限制

- 仅检索不生成答案时，必须人工判断相似度是否足以回答；
- 无 DeepSeek 密钥时 `--answer` 路径不可用（密钥只放环境变量，绝不写进文件）；
- 报告和 PPT 中的数字都可以由上述命令重新产生。
