# AudioQR

## Prepared the environment

```pip3 install -r requirements.txt ```

## Download the required datasets

```bash dataset_download.sh```

## Run Training

```python3 train.py -c configs/ljs_base.json -m AUGsimple_10-20_Dual_decoder_mixed_mel_10_20k_20k_agmt_1_2k_200 --ptb_type mixed --mel_w 10 --mel_start 20000 --mel_len 20000 --agmt_w 1 --agmt_start 2000 --agmt_len 200 --batch_size 64 --msg_dim 50 --max_step 1000000```