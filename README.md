# NLP Final: 醫病訊息決策競賽春季賽
- [Description Slide](https://drive.google.com/file/d/1s8Ks23clANaH0azIkw2GtzWmwFIEH4C7/view)
- No Kaggle competition

## Installation
- Install packages in `Pipfile`.
- Put data (csv files) in `data/`.
- setup a wandb account and modify the params in wandb.init function in `train_qa.py`

## Training
- Question answering
```bash
# ensemble
python3 train_qa.py --base_model hfl/chinese-xlnet-base --batch_size 2 --max_seq_length 512 --num_epoch 100 --ensemble True --wandb_logging True --exp_name chinese-xlnet-base-512-ensemble

# basic
python3 train_qa.py --base_model hfl/chinese-xlnet-base --batch_size 2 --max_seq_length 512 --num_epoch 100 --wandb_logging True --exp_name chinese-xlnet-base-512-ensemble
```
- The program saves the best model by the exact match of validation data (can be changed in args).
- Model will be saved in `model/`.

## Prediction
```bash
python3 predict.py --base_model <the_base_model> --checkpoint <the_checkpoint_model>
```
- Prediction will be saved in `prediction/`.

## Commandline Arguments
| Arg              	| Default               	| Help                                            	|
|------------------	|-----------------------	|-------------------------------------------------	|
| train_data       	| train.json            	| Training and validation data                    	|
| data_dir         	| data/                 	| Directory to the dataset                        	|
| base_model       	| bert-base-chinese     	| Base pre-trained language model                 	|
| model_dir        	| model/                	| Directory to save model files                   	|
| lr               	| 1e-5                  	| Learning rate                                   	|
| wd               	| 1e-2                  	| Weight decay                                    	|
| ensemble         	| False                 	| Use ensemble                                    	|
| batch_size       	| 8                     	| Batch size                                      	|
| train_val_split  	| 0.1                   	| Splitting ratio of training and validation data 	|
| max_seq_length   	| 512                   	| Maximum length of tokenizer output              	|
| device           	| cuda:0                	| Training device                                 	|
| num_epoch        	| 10                    	| Training epoch                                  	|
| n_batch_per_step 	| 2                     	| num of epoch to update optimizer                	|
| metric_for_best  	| valid_loss            	| Store best model metric                         	|
| wandb_logging    	| False                 	| Logging on wandb                                	|
| exp_name         	| bert-base-chinese-512 	| Run name on wandb                               	|


## TODO
1. Train on last checkpoint
2. Store training cofig (e.g., max_len_seq, base_model) and load in prediction