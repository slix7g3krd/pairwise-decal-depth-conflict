# pairwise-decal-depth-conflict

## 领域

游戏、图形与多媒体

## 主题

贴花投影深度冲突

## 说明

投影重建、深度偏置与动态附着。

初始状态下 `graphics/decal_depth_conflict.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
