# gozaru_wai — 2クライアント連合LoRAの例

Qwen2.5-1.5B-Instruct を LoRA で連合学習する PoC。
client0 = gozaru（語尾「でござる」, HF: bbz662bbz/databricks-dolly-15k-ja-gozaru）
client1 = wai（自作, 下記参照）

## "wai" データセットについて

kunishou/databricks-dolly-15k-ja の output に
語頭マーカー「いやはや、」を付与し、一人称を置換したもの。
名前の "wai" は一人称を「わい」化しているため。
計測（count_style.py）で追うマーカーは語頭「いやはや」。

## データ生成（jsonl はリポジトリに含めない）

```
python make_wai_dataset_full.py   # wai_full.jsonl（約15,015件）
python make_mix_dataset_full.py   # mix_full.jsonl（gozaru+wai 混合 30,030件）
```

## 実行と計測

```
python run_llm.py froseai_conf_llm.yml | tee train_rXX.log
mv fed-lora fed-lora-rXX
python count_style.py XX ./fed-lora-rXX   # results.csv に追記
```
