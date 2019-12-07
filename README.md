# 语义角色标注
本项目的主要目的是实现2015年的一篇论文[Zhou and Xu](https://www.researchgate.net/publication/283806596_End-To-end_learning_of_semantic_role_labeling_using_recurrent_neural_networks). 这篇论文提出了一种新的使用LSTM做端对端的方法, 并在当年可以达到state-of-the-art的水平. 不同于传统的SRL方法需要句法信息，文中提出的方法只需要输入原始的context, 输出SRL. 详细分析可以参考[博文](https://blog.csdn.net/m0_37722110/article/details/96202467). 代码可以在Python 3 & TensorFlow 1.2上运行.

# 模型
序列信息输入给正向的LSTM层，这个层的输出作为紧接着的反向的LSTM层的输入，这两层构成一对LSTM层，一共有4对堆叠到一起，最上面是CRF层，架构如下图:
![](https://github.com/mmichazzj/Semantic-Role-Labeling/blob/master/pics/pic1.jpg)

# 数据集
[CoNLL-2005 shared task](https://www.cs.upc.edu/~srlconll/soft.html)F1得分81.07
[CoNLL-2012 shared task](http://conll.cemantix.org/2012/data.html)F1得分81.27.
数据集的获取可以参考[这篇](https://www.jianshu.com/p/025bf2bd0ed5)博客.
