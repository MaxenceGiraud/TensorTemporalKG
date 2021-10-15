# Tensor Temporal KG

Implementation of Temporal TuckER (TuckERT and TuckERTNT models) [1] and methods based on it for temporal Knowkedge graphs completion problem.

The codebase is inspired from [TuckER's github repository](https://github.com/ibalazevic/TuckER), code from the paper of the TuckER model [2].

## Running a model 

To run a model execute the following command : 

```bash
python main.py  
                --model TuckERTTR 
                --dataset icews14 
                --n_iter 100 
                --batch_size 128 
                --learning_rate 0.001 
                --de 30
                --dr 30 
                --dt 30 
                --ranks 20 
                --device cuda 
                --early_stopping 20 
```

The follwing models are available : 
- TuckERT
- TuckERTNT 
- TuCKERTTR (TuckERT with the core tensor further decomposed using a tensor ring)
- TuckERCPD (TuckERT with the core tensor further decomposed using a CP)
- TuckERTTT (TuckERT with the core tensor further decomposed using a tensor train)

and the following datasets :
- icews14
- icews05-15

## References

[1] P. Shao, G. Yang, D. Zhang, J. Tao, F. Che, and T. Liu. Tucker decomposition-based
Temporal Knowledge Graph Completion. arXiv:2011.07751 [cs], Nov. 2020

[2] I. Balazevic, C. Allen, and T. Hospedales. TuckER: Tensor Factorization for Knowledge Graph Completion. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), 2019.