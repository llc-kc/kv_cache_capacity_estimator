[English](README.md) | [简体中文](README.zh-CN.md)

# KV Cache 容量仿真与基准测试

# 功能介绍



# 使用方法

```bash
python src/mattson_capacity_analyzer.py requests.jsonl \
  --kv-bytes-per-token 32768 --page-size 64 \
  --capacities 10GB,100GB,1TiB
```



# 结果解释

## 指标定义

```
Page hit = 命中的 page 访问数 / 全部 page 访问数

Reuse hit = 命中的可复用 page 数 / 所有非冷启动 page 访问数

Prefix hit = 各请求从开头连续命中的 page 数之和 / 全部 page 访问数
```



每次运行还会追加一项容量为 `infinite` 的结果。该结果仅检查 page key 此前是否出现过，不比较 reuse distance。JSON 输出中，这一项的 byte 容量和page 容量字段均为 `null`。

`GB` 使用十进制单位（`10^9` bytes）；二进制单位（`2^30` bytes）请使用`GiB`。
