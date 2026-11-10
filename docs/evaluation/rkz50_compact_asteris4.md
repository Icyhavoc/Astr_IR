# RKZ50 紧凑 ASTERIS4 实验分支

分支：`experiment/rkz50-compact-asteris4`。

该分支保留原作者 Restormer/ASTERIS 的时空注意力结构，但将正式默认从完整 ASTERIS8 改为紧凑 ASTERIS4：

| 配置 | 紧凑 ASTERIS4 | 完整 ASTERIS8 对照 |
|---|---:|---:|
| 时间深度 | 4 | 8 |
| `f_maps` | 8 | 24 |
| blocks | 1/1/2 | 4/6/6/8 |
| refinement blocks | 1 | 4 |
| 参数量 | 233,063 | 13,728,210 |

紧凑版参数量约为完整模型的 1.70%，即约小 59 倍。默认裁块为 64×64，每个训练序列每 epoch 抽取 256 个随机组合；这只增加已有训练天区内的覆盖，不等价于增加独立天区。

## 不变的科学约束

- 只读 background science frames，DQ/无覆盖像素不进入 loss。
- `90000002/04` 训练、`90000003` 验证、`90000005_1/2` 同属独立 test 天区。
- 从随机权重开始，不加载 JWST checkpoint。
- validation MSE 仅作诊断；正式 checkpoint 使用与输入共加相同的 matched-filter，在冻结每图误检预算和通量响应护栏下最大化盲注入完整率。
- 输出始终同时保留相同曝光的 input coadd、网络 coadd 和 residual，不能只展示网络图。
- 面向用户的图必须使用 DQ、有效覆盖和坏点表作 display-only mask，尤其屏蔽左上集中坏点，不能改写科学 FITS。

## 当前验证状态

CPU 上已完成 `1×1×4×16×16` 随机体数据的前向、论文损失和反向传播 smoke test；输出形状保持一致，loss 与梯度有限。尚未 prepare 真实数据、训练 epoch、生成 checkpoint 或运行真实测试天区推理，因此不能声称弱源检出能力改善。
