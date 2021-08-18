# Tensor Temporal KG

Temporal TuckER based method for temporal Knowkedge graphs 



## Running a model 

To run a model execute the following command : 

```bash
python main.py --model TuckERTTR --dataset icews14 --n_iter 100 --batch_size 128 --learning_rate 0.001 --de 30 --dr 30 --dt 30 --ranks 20 --cuda true --early_stopping 20 
```

The follwing models are available : 
- TuckERT
- TuCKERTTR
- TuckERCPD


and the following datasets :
- icews14
- icews05-15
