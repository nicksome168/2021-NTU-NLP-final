# NLP Final: 醫病訊息決策競賽春季賽
- [Description Slide](https://drive.google.com/file/d/1s8Ks23clANaH0azIkw2GtzWmwFIEH4C7/view)
- No Kaggle competition

## Installation
- Install packages in `Pipfile`.
- Put data (csv files) in `data/`.

## Training
- Question answering
```
python3 train_qa.py
```
- The program saves the best model by the exact match of validation data (can be changed in args).
- Model will be saved in `model/`.

## Prediction
- TODO
