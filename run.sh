pip3 install -r requirements.txt
python3 train_qa.py --base_model hfl/chinese-xlnet-base --batch_size $1 --max_seq_length 1024 --num_epoch 100 --exp_name $2 --train_data train_cn.json --n_batch_per_step 5 --wandb_logging --lr 5e-5
python3 predict.py --base_model hfl/chinese-xlnet-base --batch_size $1 --max_seq_length 1024 --checkpoint model/$2.pt --data_path data/public_cn.json