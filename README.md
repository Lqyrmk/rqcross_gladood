# new idea of GLAD and GLOD

“换模块大王”

## Latest Experimental Progress 最新实验进展

图级异常检测数据集很难 work ，依旧

* MMoE 有用但又不太有用
* Prototype Learning 得优化一下
* Cross-Attention + Prototype 效果还行
* Self-Attention 效果不多
  * 和 Cross-Attention 一块用的话，会影响 Cross-Attention 的 Attention 热力图，学不到太多东西
* 看起来 graph-level 的 loss 很好用，加上 node-level 的 loss 容易掉点，但如果一开始就直接 pooling 到 graph-level，粒度太粗，效果不行。正确做法是开始先学好 node-level 的 embedding，然后再 pooling 到 graph-level 上做对比，这个作用挺大的
* Self-Attention 进展：
  * 本身好像没啥提升，甚至负作用。
  * 加了个图内 mask 后，试了下感觉应该还是有一点提升的，确保捕获到的信息在同一图内。但是都挺玄学的，有时候又不如不加。self-attention (no mask) < self-attention (intra mask) < no self-attention
* 去掉了 loss_pp 即原型间对比，发现提升了，看起来没多大用