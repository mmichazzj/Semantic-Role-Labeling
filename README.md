# Semantic-Role-Labeling
本项目的主要目的是实现2015年的一篇论文https://hmjwhmjw.github.io/src/2019-3-27-related-papers/E.pdf. 这篇百度的论文提出了一种新的使用LSTM做端对端的方法, 并在当年可以达到state-of-the-art的水平. 不同于传统的SRL方法需要句法信息，文中提出的方法只需要输入原始的context, 输出SRL. 详细分析可以参考博文https://blog.csdn.net/m0_37722110/article/details/96202467. 代码可以在Python 3 & TensorFlow 1.2上运行.
# Model
模型架构如下图:
