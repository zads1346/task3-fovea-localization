# 2026task3 项目文件总索引

统一目录：`D:\KaggleProjects\2026task3`

迁移完成：2026-10-31T15:09:23.618185+08:00。共 **20,001 个既有文件**，合计 **21.791 GB**。迁移生成的脚本、索引和报告未计入此数。

## 目录地址

| 内容 | 新地址 |
|---|---|
| 完整项目 | `D:\KaggleProjects\2026task3\project` |
| 训练代码、实验与解压数据 | `D:\KaggleProjects\2026task3\project\work` |
| Notebook、模型权重、各版本预测与报告 | `D:\KaggleProjects\2026task3\project\outputs` |
| 历史备份压缩包 | `D:\KaggleProjects\2026task3\project\backups` |
| 参考计分与人工调整诊断文件 | `D:\KaggleProjects\2026task3\reference_audit` |
| 原始数据包、源 Notebook 和上传日志 | `D:\KaggleProjects\2026task3\attachments\downloads` |
| 微信来源的参考资料、CSV 和版本压缩包 | `D:\KaggleProjects\2026task3\attachments\wechat` |
| 全量文件地址、SHA256 与迁移记录 | `D:\KaggleProjects\2026task3\migration` |

## 关键文件

- 最近的纯模型提交：`project\outputs\wide_refine_20261027\submission.csv`
- 8.2 基线提交：`project\outputs\restored_20261027\submission_candidate.csv`
- 双尺度模型提交：`project\outputs\dual_context_20261027\submission.csv`
- 双尺度 Kaggle Notebook：`project\outputs\dual_context_20261027\fovea_dual_context_kaggle.ipynb`
- 原始数据包：`attachments\downloads\2026task3.zip`
- 用户源 Notebook：`attachments\downloads\notebook3 (1).ipynb`
- 人工16点多参考分版本：`reference_audit\reference_target16x_wide_20261027\submission.csv`。此文件依参考标签定向修改，不是独立模型成绩。

以上相对地址均从本索引所在的统一目录开始。

## C 盘清理与旧路径

已对 C/D 两份逐文件计算 SHA256，并在删除前再次核验。已清理 C 盘原始数据副本 **21.731 GB**（逻辑文件大小）。C 盘可用空间从预检时的 **1.01 GB** 增至 **22.57 GB**；系统后台写入可能影响差值。

原主项目根目录被当前 Codex 应用占用，因此保留空容器及 `backups`、`outputs`、`work` 三个目录链接；原评分目录也是目录链接。它们指向 D 盘，并非 C 盘残留的数据副本。后续新建文件、打开项目请使用 D 盘的新根目录。

微信和原下载位置本来就在 D 盘，归档后在旧位置保留同卷硬链接，不会另占一份文件空间。模型、数据、代码、CSV 和历史报告内容均未改写，因此冻结校验值保持有效。Python/Conda、Codex 等共享软件环境不属于本项目文件，本次未迁移。

## 完整清单与已知缺失

`migration\all_project_files.csv` 包含全部 20,001 个文件的旧地址、新地址、大小与 SHA256。`migration\migration_complete.json` 记录清理和空间核验结果。

历史上传的5个 Downloads 文件已在 `D:\下载10.25` 找到并统一归档。两张历史 `codex-clipboard-*.png` 截图在原 Temp 路径已不存在，无法从该位置迁移；具体旧地址见完成报告。原可视化目录为空，无可搬迁文件。
