convert_longbench_v2_to_openai.py can be used to convert json download from https://modelscope.cn/datasets/ZhipuAI/LongBench-v2/files to openai format file.

Then you can test this cache capacity estimator.

You can also test sglang engine by evalscope like:

```
evalscope perf \
  --dataset line_by_line \
  --dataset-path data_short.openai.jsonl \
  --url http://localhost:30000/v1/chat/completions \
  --model GLM-5.2 \
  --max-tokens 1 \
  --min-tokens 1 \
  --parallel 10 \
  --number 180
```

