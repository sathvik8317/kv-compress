# Results

20 needle-in-a-haystack prompts (2k–16k tokens, 5 depths) and 120 LongBench-E prompts (4k–8k words, middle-truncated to 8k tokens). Qwen2.5-0.5B-Instruct, fp16, greedy. Memory = stored KV bytes of the prompt / full fp16 (quantized: payload + scales + zeros).

| variant | KV memory | NIAH acc | LongBench | agreement w/ full | K rel. err |
|---|---:|---:|---:|---:|---:|
| full | 100.0% | 100% | 40.8 | 100% | - |
| int8 | 51.6% | 100% | 40.1 | 85% | 0.9% |
| int4-kivi | 31.4% | 100% | 39.5 | 83% | 3.2% |
| int4-token | 31.2% | 10% | 28.4 | 28% | 9.6% |
| int2-kivi | 18.9% | 60% | 27.5 | 45% | 16.4% |
| snapkv+int4-kivi@0.5 | 15.7% | 80% | 40.9 | 75% | 3.4% |
| snapkv+int4-kivi@0.25 | 7.9% | 70% | 39.4 | 68% | 3.5% |
| snapkv@0.5 | 50.0% | 80% | 40.9 | 81% | - |
| snapkv@0.25 | 25.0% | 75% | 39.5 | 73% | - |
| snapkv@0.1 | 10.0% | 45% | 38.5 | 52% | - |
| snapkv@0.05 | 5.0% | 30% | 35.6 | 43% | - |
| h2o@0.5 | 50.0% | 20% | 40.6 | 69% | - |
| h2o@0.25 | 25.0% | 20% | 36.4 | 55% | - |
| h2o@0.1 | 10.0% | 20% | 34.2 | 42% | - |
| h2o@0.05 | 5.0% | 20% | 27.5 | 34% | - |
| streaming@0.5 | 50.0% | 40% | 37.3 | 56% | - |
| streaming@0.25 | 25.0% | 20% | 35.3 | 44% | - |
| streaming@0.1 | 10.0% | 20% | 28.1 | 35% | - |
| streaming@0.05 | 5.0% | 20% | 23.0 | 34% | - |

## LongBench per task (score × 100)

| variant | 2wikimqa | hotpotqa | lcc | multifieldqa_en | trec | triviaqa |
|---|---:|---:|---:|---:|---:|---:|
| full | 8.5 | 19.6 | 44.9 | 41.4 | 50.0 | 80.1 |
| int8 | 8.5 | 23.4 | 37.0 | 41.4 | 50.0 | 80.1 |
| int4-kivi | 8.9 | 19.5 | 42.6 | 38.3 | 50.0 | 77.6 |
| int4-token | 15.0 | 20.4 | 25.8 | 24.8 | 30.0 | 54.7 |
| int2-kivi | 2.2 | 17.0 | 29.3 | 24.5 | 40.0 | 51.9 |
| snapkv+int4-kivi@0.5 | 8.5 | 23.4 | 38.5 | 39.9 | 55.0 | 80.1 |
| snapkv+int4-kivi@0.25 | 7.7 | 22.2 | 38.7 | 35.2 | 55.0 | 77.6 |
| snapkv@0.5 | 8.5 | 22.2 | 45.0 | 39.7 | 50.0 | 80.1 |
| snapkv@0.25 | 8.1 | 21.7 | 45.5 | 36.7 | 50.0 | 75.1 |
| snapkv@0.1 | 5.7 | 20.9 | 37.7 | 39.4 | 50.0 | 77.1 |
| snapkv@0.05 | 11.4 | 17.2 | 39.0 | 29.2 | 45.0 | 71.9 |
| h2o@0.5 | 7.3 | 17.6 | 47.0 | 36.6 | 55.0 | 80.1 |
| h2o@0.25 | 5.2 | 15.3 | 40.0 | 36.1 | 55.0 | 66.9 |
| h2o@0.1 | 9.0 | 13.8 | 36.1 | 31.3 | 45.0 | 69.8 |
| h2o@0.05 | 9.7 | 11.9 | 32.6 | 25.2 | 40.0 | 45.5 |
| streaming@0.5 | 6.3 | 16.0 | 41.9 | 24.6 | 55.0 | 80.1 |
| streaming@0.25 | 11.3 | 10.2 | 43.5 | 25.5 | 45.0 | 76.1 |
| streaming@0.1 | 8.8 | 5.9 | 42.3 | 23.7 | 40.0 | 48.1 |
| streaming@0.05 | 10.9 | 8.8 | 41.0 | 21.6 | 20.0 | 35.6 |

## Needle accuracy by context length

| variant | 2k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| full | 100% | 100% | 100% | 100% |
| int8 | 100% | 100% | 100% | 100% |
| int4-kivi | 100% | 100% | 100% | 100% |
| int4-token | 0% | 20% | 20% | 0% |
| int2-kivi | 60% | 80% | 80% | 20% |
| snapkv+int4-kivi@0.5 | 60% | 80% | 100% | 80% |
| snapkv+int4-kivi@0.25 | 60% | 80% | 80% | 60% |
| snapkv@0.5 | 60% | 80% | 100% | 80% |
| snapkv@0.25 | 60% | 100% | 80% | 60% |
| snapkv@0.1 | 40% | 60% | 60% | 20% |
| snapkv@0.05 | 20% | 20% | 40% | 40% |
| h2o@0.5 | 20% | 20% | 20% | 20% |
| h2o@0.25 | 20% | 20% | 20% | 20% |
| h2o@0.1 | 20% | 20% | 20% | 20% |
| h2o@0.05 | 20% | 20% | 20% | 20% |
| streaming@0.5 | 40% | 40% | 40% | 40% |
| streaming@0.25 | 20% | 20% | 20% | 20% |
| streaming@0.1 | 20% | 20% | 20% | 20% |
| streaming@0.05 | 20% | 20% | 20% | 20% |

Attention sparsity (observation-window queries, mean over layers and prompts): top 1% of keys hold 77% of attention mass, top 5% hold 88%, the first 4 tokens hold 32%.
