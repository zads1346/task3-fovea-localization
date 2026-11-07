2026task3 Kaggle科研实验包 v2.0

1. 导入 fovea_research_kaggle.ipynb 到 Kaggle Notebook。
2. Add Input 添加2026task3比赛数据。
3. 将 fovea_pretrained_weights.zip 上传为一个Kaggle Dataset，挂载到Notebook。
   ZIP自动解压或保留ZIP均可，程序会自动查找并校验。
4. 启用GPU，Internet可关闭，保持SMOKE_TEST=False，Run All。
5. /kaggle/working/submission.csv 为正式运行输出，默认B4。

文件：
  fovea_research_kaggle.ipynb  完整Notebook，代码全部内嵌
  fovea_research_kaggle.py     同等功能Python脚本
  fovea_pretrained_weights.zip  官方ResNet18 ImageNet-1K权重及来源清单
  research_validation.json   本地实际验收范围（若包含），不是正式比赛成绩

不需要同时把.py上传到Kaggle；Notebook可独立运行。
权重是ImageNet初始化，不是训练完成的中心凹预测模型。
默认5个外层折，每折7个训练任务，共35次；消融共享粗/细模型。
每折校准数据始终不参与最终网络训练；不能使用测试标签调整坐标。
SMOKE_TEST只做低分辨率1轮流程检查，输出submission_SMOKE_TEST.csv。
完整正式五折训练需由Kaggle GPU执行；本地验收不代表精度。
仅支持当前80张均可见的训练数据；不存在可靠的不可见类学习证据。
复用只覆盖已完成的外层折，不是中断epoch/优化器恢复。
详细方法、假设、消融和输出清单见Notebook说明。
