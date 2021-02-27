# 模型蒸馏踩坑记录
以下经验来源于token-classification任务，主要针对bert-alike模型，迁移到其他场景时不一定有用
涉及到的可调节部分

[模型相关](#模型相关)  
[蒸馏参数调节规律](#蒸馏参数调节规律)  
[蒸馏实验记录](#蒸馏实验记录)

## 模型相关
1. 模型的骨架：尽量选取大参数量的模型骨架，一般情况下参数量越多，在经过合适训练后，其上限也会越高
2. 模型的初始化方式
    * fine-tune后的模型权重：一般而言，在fine-tune效果比较好的时候，用这个比较合适，且可以直接使用全部的参数。但跨任务时效果无保证
    * 预训练模型权重： 仅可以利用共享的transformer层，但迁移的知识比较普适，一般都可以当作baseline  
    *简而言之，用fine-tune模型的权重初始化最大收益大，用预训练模型权重较稳定*
3. 模型的训练参数：与常规fine-tune无区别，用同样的思路选取learning-rate, weight decay, batch size即可
## 蒸馏参数调节规律
默认蒸馏loss由teacher loss，student loss 和intermediate layers loss三部分组成 
1. 温度常数T调节
    由于T本质起到的作用是平滑logits，温度常数越大，logits越平滑，老师模型对最后loss的影响就越小，但学生模型的训练效果 
    与温度常数的关系应该是凸函数，存在一个极值使得蒸馏效果达到最好
2. 中间层跟踪
    * 中间层形状一样：个人经验中间层具体哪儿对哪个无所谓，只要满足头对头尾对尾  
        举两个例子
        ```
            # T: Roberta(12 layers)  S: Roberta-tiny(4-layers)
            layer_T: 0               layer_S: 0
            layer_T: 4               layer_S: 1
            layer_T: 8               layer_S: 2
            layer_T: 11              layer_S: 3
            
            # T: Roberta-large(24 layers)  S: Roberta(12-layers)
            layer_T: 0               layer_S: 0
            layer_T: 2               layer_S: 1
            .......
            layer_T: 20              layer_S: 10
            layer_T: 23              layer_S: 11
        ```
        除了这样一层对一层，也可以多层对一层
    * 中间层形状不一样：用一维的卷积核把老师模型的中间层降维即可（参考华为tinybert操作） 
    
![Image text](https://github.com/another1s/Distill/blob/master/%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F%E8%B0%83%E5%8F%82%E4%BF%AE%E6%94%B9%E6%96%B9%E5%90%91.png)
## 蒸馏实验记录

|          模型的参数选取                             |       蒸馏设定       |         初始化方式                       |  产出模型表现     | 
|:-------------------------------------------------:|:-------------------:|:--------------------------------------:|:---------------:|
|  T:Electra(24 layers), S:tinyElectra(12 layers)   |        10           |       预训练权重                         |                 |
|  T:Roberta-large(24 layers), S: Roberta(12 layers)|        10           |       预训练权重                         |                 |
|  T:Roberta-large(24 layers), S: Roberta(12 layers)|        10           |       多任务模型的fine-tune权重           |                 |
|  T:Roberta-large(24 layers), S: Roberta(6 layers) |        15           |       预训练权重                         |                 |
|  T:Roberta-large(12 layers), S: Roberta(4 layers) |        15           |       多任务模型的fine-tune权重           |                 |