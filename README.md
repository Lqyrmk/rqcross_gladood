# new idea of GLAD and GLOD

“换模块大王”

## Latest Experimental Progress 最新实验进展

图级异常检测数据集很难 work ，依旧

* MMoE 有用但又不太有用
* Prototype Learning 得优化一下
* Cross-Attention + Prototype 效果还行
* Self-Attention 效果不多
  * 和 Cross-Attention 一块用的话，会影响 Cross-Attention 的 Attention 热力图，学不到太多东西
* 看起来 graph-level 的 loss 很好用，加上 node-level 的 loss 容易掉点，但如果一开始就直接 pooling 到 graph-level，粒度太粗，效果不行。正确做法是开始先学好 node-level 的 embedding，然后再 pooling 到 graph-level 上做对比