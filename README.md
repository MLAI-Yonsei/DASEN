# Data Adaptive Stochastic Ensemble Net: Optimizing Infection Predictions for COVID-19 Cluster Analysis

>This repository contains PyTorch implemenations of "Data Adaptive Stochastic Ensemble Net: Optimizing Infection Predictions for COVID-19 Cluster Analysis".
>> [Sungjun Lim](https://sungjun98.github.io/), Yongtaek Lim, Hojun Park,Junggu Lee, Jaehun Jung, [Kyungwoo Song](https://mlai.yonsei.ac.kr/)

------

</br>

## Abstract
![scheme](https://github.com/user-attachments/assets/5d8fa144-44ae-45fd-a75a-ea0e414e6298)

> * Data Adaptive Stochastic Ensemble Net (DASEN) is Dirichlet-based ensemble technique, which incorporates discrete characteristics of diverse models and shows robust infection prediction in realistic settings.

------

</br>

## Setup

```
git clone https://github.com/MLAI-Yonsei/DASEN.git
cd DASEN

# Create and activate a conda environment
conda create -y -n dasen python=3.9.15
conda activate dasen

# install packages
pip install -r requirements.txt
```

------


</br>

## Run
We explain the code snippet for DASEN. Note that DASEN requires pre-trained models to ensemble them in stochastic method.

The `run_baseline.py` allows to run all baselines, Linear Regression (LR), Ridge Regression (RR), Multi-layer Perceptron (MLP), Graph Convolution Network (GCN), Graph Attention Network (GAT), and MagNet.

The `deep_ensemble.py` allows to run Deep Ensemble with pre-trained models.

The `adaptive_ensemble.py` allows to run adaptive Ensemble with pre-trained models.

The `DASEN.py` allows to run DASEN with pre-trained models.

#### Linear Regression (LR)
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=Linear --hidden_dim={HIDDEN_DIM} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT}
```

* ```TARGET_ORDER``` &mdash; the contact order in which predict the number of confirmed cases (2, 3, 4)
* ```HIDDEN_DIM``` &mdash; number of predictor
* ```EPOCHS``` &mdash; number of epochs
* ```LR_INIT``` &mdash; initial learning rate
* ```WD``` &mdash; weight decay
* ```DROP_OUT``` &mdash; dropout rate


#### Ridge Regression (RR)
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=Ridge --hidden_dim={HIDDEN_DIM} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT} \
--lamb={LAMBDA}
```

* ```LAMBDA``` &mdash; Penalty term for Ridge Regression


#### Multi-layer Perceptron (MLP)
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=MLP --hidden_dim={HIDDEN_DIM}  --num_layers={NUM_LAYERS} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT}
```

* ```NUM_LAYERS``` &mdash; number of layers


#### Graph Convolution Network (GCN)
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=GCN --hidden_dim={HIDDEN_DIM} --num_layers={NUM_LAYERS} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT}
```


#### Graph Attention Network (GAT)
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=GAT --hidden_dim={HIDDEN_DIM} --num_layers={NUM_LAYERS} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT}
```


#### MagNet
```
run_baseline.py --target_order={TARGET_ORDER} \
--model=MagNet --hidden_dim={HIDDEN_DIM} --epochs=100 \
--lr_init={LR_INIT} --wd={WD} --drop_out={DROP_OUT} \
--q={Q} [--complex_activation] [--trainable_q]
```

* ```Q``` &mdash; ratio of directional flow 
* ```--complex_activation``` &mdash; use complex activation
* ```--trainable_q``` &mdash; set trainable q


</br>

### Ensembles
For ensemble methods, we assume we save the pre-trained models in path, as follow
```
LR : /{target_order}-{seed}/lr/lr-best_tr_pred_cv_tr{fold}.csv'
RR : /{target_order}-{seed}/rr/rr-best_tr_pred_cv_tr{fold}.csv'
MLP : /{target_order}-{seed}/mlp/mlp-best_tr_pred_cv_tr{fold}.csv'
GCN : /{target_order}-{seed}/gcn/gcn-best_tr_pred_cv_tr{fold}.csv'
GAT : /{target_order}-{seed}/gat/gat-best_tr_pred_cv_tr{fold}.csv'
MagNet : /{target_order}-{seed}/mag/mag-best_tr_pred_cv_tr{fold}.csv'
```
* ```target_order``` &mdash; the contact order in which predict the number of confirmed cases (2, 3, 4)
* ```seed``` &mdash; random seed (1000, 1001, 1002, 1003, 1004)
* ```fold``` &mdash; fold for cross validation (1, 2, 3)



#### Deep Ensemble (DE)
```
deep_ensemble.py --target_order={TARGET_ORDER} 
```

#### Adaptive Ensemble (AE) - LR
```
adaptive_ensemble.py --target_order={TARGET_ORDER} \
--method=lr --lr_init={LR_INIT} --wd={WD} --epoch=200
```

#### Adaptive Ensemble (AE) - SVR
```
adaptive_ensemble.py --target_order={TARGET_ORDER} \
--method=svr --epoch=200 --eps={EPS}
```

* ```EPS``` &mdash; epsilon in SVR loss

</br>

#### Data Adaptive Stochastic Ensemble Net (DASEN)
```
dasen.py --target_order={TARGET_ORDER} \
--lr_init={LR_INIT} --wd={WD} --epoch=200
```

---------

</br>

## Contact
For any questions, discussions, and proposals, please contact to `lsj9862@yonsei.ac.kr` or `kyungwoo.song@gmail.com`