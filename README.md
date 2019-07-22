# 语义角色标注
本项目的主要目的是实现2015年的一篇论文https://hmjwhmjw.github.io/src/2019-3-27-related-papers/E.pdf. 这篇百度的论文提出了一种新的使用LSTM做端对端的方法, 并在当年可以达到state-of-the-art的水平. 不同于传统的SRL方法需要句法信息，文中提出的方法只需要输入原始的context, 输出SRL. 详细分析可以参考博文https://blog.csdn.net/m0_37722110/article/details/96202467. 代码可以在Python 3 & TensorFlow 1.2上运行.

# 模型
序列信息输入给正向的LSTM层，这个层的输出作为紧接着的反向的LSTM层的输入，这两层构成一对LSTM层，一共有4对堆叠到一起，最上面是CRF层，架构如下图:
![](https://github.com/mmichazzj/Semantic-Role-Labeling/blob/master/pics/pic1.jpg)
