import wandb
import torch
import torch.nn as nn
from numpy.random import dirichlet
import torch.distributions as dist
import pickle
import utils
import numpy as np
import pandas as pd
import os, time, tabulate

import argparse
import warnings
warnings.filterwarnings('ignore')
# --------------------------------------------------    

## Argparse ------------------------------------------------------
parser = argparse.ArgumentParser(description="")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--target_order", type=int, default=3,
          help="Decide how much order we get (Default : 3)") # 3: 3차까지는 주어진 상태에서, 4차 여부 예측

parser.add_argument("--method", type=str, default='deep_ensemble',
                    choices=['deep_ensemble'])

parser.add_argument("--fold", type=int, default=1,
        help="tr fold (Default: 1)")

parser.add_argument("--result_path", type=str, default='./best_result', # required=True,
    help="path to load saved model to resume training",)

parser.add_argument("--graph_num_path", type=str, default='./graph_nums',
        help="path to load graph num")

parser.add_argument("--save_path", type=str, default="./exp_result",
            help="Path to save best model dict")

parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

parser.add_argument("--wandb_entity", type=str, default="mlai_medical_ai",
        help="name of entity for wandb")

parser.add_argument("--wandb_name", type=str, default="dgnn_v2.5",
        help="name of project for wandb")

parser.add_argument("--rm_feature", type=int, default=-999,
        help="removed feature (0 to 16)(Default : -999 (=None))")

# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./data/20221122_raw_data_utf-8.csv',
    help="path to datasets location",)

parser.add_argument("--replace_missing_value", type=str, default='replace',
    choices=['replace', 'mice', 'binomial'])

parser.add_argument("--birthyear_rule", type=str, default="mdlp", choices=['birthyear_chunk_id', 'mdlp'],
    help="birthyear_chunk_id: divided by 30 years, mdlp: Minimum Description Length Binning")
#----------------------------------------------------------------

# Criterion -----------------------------------------------------
parser.add_argument(
    "--criterion",
    type=str, default='MSE', choices=["MSE", "RMSE"],
    help="Criterion for training (default : MSE)")

parser.add_argument(
    "--eval_criterion",
    type=str, default='MAE', choices=["MAE", "RMSE"],
    help="Criterion for training (default : MAE)")
#----------------------------------------------------------------


# Ensemble params ---------------------------------------------------------
parser.add_argument("--only_gnn", action="store_true",
        help="ensemble only for GNN models (GCN, GAT, MagNet) (Default : False)")

args = parser.parse_args()
#----------------------------------------------------------------

## Set seed, save path, and device ----------------------------------------------------------------
utils.set_seed(args.seed)
save_path = args.save_path + f'/{args.target_order}-{args.seed}/{args.method}'
if args.rm_feature != -999:
    save_path = save_path + f'/rm{args.rm_feature}'
os.makedirs(save_path, exist_ok=True)

# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Seed : {args.seed} / Order : {args.target_order} / Fold : {args.fold}")  # / Device : {args.device}")
#-------------------------------------------------------------------------------------

## wandb init----------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(project=args.wandb_name, entity=args.wandb_entity)
    wandb.config.update(args)
    run_name = f"{args.method}-order{args.target_order}-seed{args.seed}-fold{args.fold}"
    if args.rm_feature != -999:
        run_name + f"-rm_feature({args.rm_feature})"
    wandb.run.name = run_name
    

## Load Graph Index ------------------------------------------------------------
os.makedirs(args.graph_num_path, exist_ok=True)
args.tr_graph_num_path = args.graph_num_path + f'/tr-{args.seed}-{args.target_order}_cv_tr{args.fold}.pkl'
args.val_graph_num_path = args.graph_num_path + f'/val-{args.seed}-{args.target_order}_cv_tr{args.fold}.pkl'
args.te_graph_num_path = args.graph_num_path +  f'/te-{args.seed}-{args.target_order}_cv_tr{args.fold}.pkl'

if not args.tr_graph_num_path in os.listdir(args.graph_num_path):
    import utils
    import pickle
    import itertools
    def get_graph_num(seed, order, fold):
        utils.set_seed(seed)
        final_data_list, _, _, _ = utils.full_load_data_cv(data_path = args.data_path,
                                            num_features = 17,
                                            target_order = order,
                                            num_folds = 3,
                                            tr_fold = fold-1,
                                            classification = True,
                                            device = 'cpu',
                                            model_name = 'MagNet',
                                            args = args)    
    
        tr_graph_idx_list = []; val_graph_idx_list = []; te_graph_idx_list = []
        for iter, data in enumerate(final_data_list):
            tr_num = sum(data[3])
            tr_graph_idx_list.append([iter] * tr_num)
            val_num = sum(data[4])
            val_graph_idx_list.append([iter] * val_num)
            te_num = sum(data[5])
            te_graph_idx_list.append([iter] * te_num)
        
        tr_graph_idx_list = list(itertools.chain.from_iterable(tr_graph_idx_list))
        with open(args.tr_graph_num_path, 'wb') as fp:
            pickle.dump(tr_graph_idx_list, fp)
            
        val_graph_idx_list = list(itertools.chain.from_iterable(val_graph_idx_list))
        with open(args.val_graph_num_path, 'wb') as fp:
            pickle.dump(val_graph_idx_list, fp)
            
        te_graph_idx_list = list(itertools.chain.from_iterable(te_graph_idx_list))
        with open(args.te_graph_num_path, 'wb') as fp:
            pickle.dump(te_graph_idx_list, fp)

    get_graph_num(args.seed, args.target_order, args.fold)
        
                
with open(args.tr_graph_num_path, 'rb') as fp:
    tr_graph_idx_list = pickle.load(fp)
tr_df = pd.DataFrame(tr_graph_idx_list, columns=['graph_idx'])

with open(args.val_graph_num_path, 'rb') as fp:
    val_graph_idx_list = pickle.load(fp)
val_df = pd.DataFrame(val_graph_idx_list, columns=['graph_idx'])

with open(args.te_graph_num_path, 'rb') as fp:
    te_graph_idx_list = pickle.load(fp)
te_df = pd.DataFrame(te_graph_idx_list, columns=['graph_idx'])


## Load Predictions ------------------------------------------------------------------------------
if args.rm_feature == -999:
    # Train Prediction paths
    lr_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/lr-best_tr_pred_cv_tr{args.fold}.csv'; lr_tr = pd.read_csv(lr_tr_path, index_col=0)
    rr_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rr-best_tr_pred_cv_tr{args.fold}.csv'; rr_tr = pd.read_csv(rr_tr_path, index_col=0)
    mlp_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/mlp-best_tr_pred_cv_tr{args.fold}.csv'; mlp_tr = pd.read_csv(mlp_tr_path, index_col=0)
    gcn_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/gcn-best_tr_pred_cv_tr{args.fold}.csv'; gcn_tr = pd.read_csv(gcn_tr_path, index_col=0)
    gat_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/gat-best_tr_pred_cv_tr{args.fold}.csv'; gat_tr = pd.read_csv(gat_tr_path, index_col=0)
    mag_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/mag-best_tr_pred_cv_tr{args.fold}.csv'; mag_tr = pd.read_csv(mag_tr_path, index_col=0)


    # Validation Prediction paths
    lr_val_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/lr-best_val_pred_cv_tr{args.fold}.csv'; lr_val = pd.read_csv(lr_val_path, index_col=0)
    rr_val_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rr-best_val_pred_cv_tr{args.fold}.csv'; rr_val = pd.read_csv(rr_val_path, index_col=0)
    mlp_val_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/mlp-best_val_pred_cv_tr{args.fold}.csv'; mlp_val = pd.read_csv(mlp_val_path, index_col=0)
    gcn_val_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/gcn-best_val_pred_cv_tr{args.fold}.csv'; gcn_val = pd.read_csv(gcn_val_path, index_col=0)
    gat_val_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/gat-best_val_pred_cv_tr{args.fold}.csv'; gat_val = pd.read_csv(gat_val_path, index_col=0)
    mag_val_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/mag-best_val_pred_cv_tr{args.fold}.csv'; mag_val = pd.read_csv(mag_val_path, index_col=0)

    # Test Prediction paths
    lr_te_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/lr-best_te_pred_cv_tr{args.fold}.csv'; lr_te = pd.read_csv(lr_te_path, index_col=0)
    rr_te_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rr-best_te_pred_cv_tr{args.fold}.csv'; rr_te = pd.read_csv(rr_te_path, index_col=0)
    mlp_te_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/mlp-best_te_pred_cv_tr{args.fold}.csv'; mlp_te = pd.read_csv(mlp_te_path, index_col=0)
    gcn_te_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/gcn-best_te_pred_cv_tr{args.fold}.csv'; gcn_te = pd.read_csv(gcn_te_path, index_col=0)
    gat_te_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/gat-best_te_pred_cv_tr{args.fold}.csv'; gat_te = pd.read_csv(gat_te_path, index_col=0)
    mag_te_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/mag-best_te_pred_cv_tr{args.fold}.csv'; mag_te = pd.read_csv(mag_te_path, index_col=0)
    
else:
    # Train Prediction paths
    lr_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/rm{args.rm_feature}/lr-best_tr_pred_cv_tr{args.fold}.csv'; lr_tr = pd.read_csv(lr_tr_path, index_col=0)
    rr_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rm{args.rm_feature}/rr-best_tr_pred_cv_tr{args.fold}.csv'; rr_tr = pd.read_csv(rr_tr_path, index_col=0)
    mlp_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/rm{args.rm_feature}/mlp-best_tr_pred_cv_tr{args.fold}.csv'; mlp_tr = pd.read_csv(mlp_tr_path, index_col=0)
    gcn_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/rm{args.rm_feature}/gcn-best_tr_pred_cv_tr{args.fold}.csv'; gcn_tr = pd.read_csv(gcn_tr_path, index_col=0)
    gat_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/rm{args.rm_feature}/gat-best_tr_pred_cv_tr{args.fold}.csv'; gat_tr = pd.read_csv(gat_tr_path, index_col=0)
    mag_tr_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/rm{args.rm_feature}/mag-best_tr_pred_cv_tr{args.fold}.csv'; mag_tr = pd.read_csv(mag_tr_path, index_col=0)

    # Validation Prediction paths
    lr_val_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/rm{args.rm_feature}/lr-best_val_pred_cv_tr{args.fold}.csv'; lr_val = pd.read_csv(lr_val_path, index_col=0)
    rr_val_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rm{args.rm_feature}/rr-best_val_pred_cv_tr{args.fold}.csv'; rr_val = pd.read_csv(rr_val_path, index_col=0)
    mlp_val_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/rm{args.rm_feature}/mlp-best_val_pred_cv_tr{args.fold}.csv'; mlp_val = pd.read_csv(mlp_val_path, index_col=0)
    gcn_val_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/rm{args.rm_feature}/gcn-best_val_pred_cv_tr{args.fold}.csv'; gcn_val = pd.read_csv(gcn_val_path, index_col=0)
    gat_val_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/rm{args.rm_feature}/gat-best_val_pred_cv_tr{args.fold}.csv'; gat_val = pd.read_csv(gat_val_path, index_col=0)
    mag_val_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/rm{args.rm_feature}/mag-best_val_pred_cv_tr{args.fold}.csv'; mag_val = pd.read_csv(mag_val_path, index_col=0)

    # Test Prediction paths
    lr_te_path = args.result_path + f'/{args.target_order}-{args.seed}/lr/rm{args.rm_feature}/lr-best_te_pred_cv_tr{args.fold}.csv'; lr_te = pd.read_csv(lr_te_path, index_col=0)
    rr_te_path = args.result_path + f'/{args.target_order}-{args.seed}/rr/rm{args.rm_feature}/rr-best_te_pred_cv_tr{args.fold}.csv'; rr_te = pd.read_csv(rr_te_path, index_col=0)
    mlp_te_path = args.result_path + f'/{args.target_order}-{args.seed}/mlp/rm{args.rm_feature}/mlp-best_te_pred_cv_tr{args.fold}.csv'; mlp_te = pd.read_csv(mlp_te_path, index_col=0)
    gcn_te_path = args.result_path + f'/{args.target_order}-{args.seed}/gcn/rm{args.rm_feature}/gcn-best_te_pred_cv_tr{args.fold}.csv'; gcn_te = pd.read_csv(gcn_te_path, index_col=0)
    gat_te_path = args.result_path + f'/{args.target_order}-{args.seed}/gat/rm{args.rm_feature}/gat-best_te_pred_cv_tr{args.fold}.csv'; gat_te = pd.read_csv(gat_te_path, index_col=0)
    mag_te_path = args.result_path + f'/{args.target_order}-{args.seed}/mag/rm{args.rm_feature}/mag-best_te_pred_cv_tr{args.fold}.csv'; mag_te = pd.read_csv(mag_te_path, index_col=0)


## Graph Index, Ground Truth, Prediction Concat
tr_df = pd.concat([tr_df, lr_tr['tr_ground_truth'], lr_tr['tr_pred'], rr_tr['tr_pred'], mlp_tr['tr_pred'], gcn_tr['tr_pred'], gat_tr['tr_pred'], mag_tr['tr_pred'], lr_tr['num_node'], lr_tr['num_edge'], lr_tr['num_missing_value_per_node'], lr_tr['num_tr'], lr_tr['num_val'], lr_tr['num_te'], lr_tr['order_list']], axis=1)
tr_df.columns = ['graph_idx', 'tr_gt', 'lr_pred', 'rr_pred', 'mlp_pred', 'gcn_pred', 'gat_pred', 'mag_pred',
                'num_node', 'num_edge', 'num_missing_value_per_node', 'num_tr', 'num_val', 'num_te', 'order_list']

val_df = pd.concat([val_df, lr_val['val_ground_truth'], lr_val['val_pred'], rr_val['val_pred'], mlp_val['val_pred'], gcn_val['val_pred'], gat_val['val_pred'], mag_val['val_pred'],lr_val['num_node'],lr_val['num_edge'], lr_val['num_missing_value_per_node'], lr_val['num_tr'], lr_val['num_val'], lr_val['num_te'], lr_val['order_list']], axis=1)
val_df.columns = ['graph_idx', 'val_gt', 'lr_pred', 'rr_pred', 'mlp_pred', 'gcn_pred', 'gat_pred', 'mag_pred',
                'num_node', 'num_edge', 'num_missing_value_per_node', 'num_tr', 'num_val', 'num_te', 'order_list']

te_df = pd.concat([te_df, lr_te['te_ground_truth'], lr_te['te_pred'], rr_te['te_pred'], mlp_te['te_pred'], gcn_te['te_pred'], gat_te['te_pred'], mag_te['te_pred'],lr_te['num_node'],lr_te['num_edge'], lr_te['num_missing_value_per_node'], lr_te['num_tr'], lr_te['num_val'], lr_te['num_te'], lr_te['order_list']], axis=1)
te_df.columns = ['graph_idx', 'te_gt', 'lr_pred', 'rr_pred', 'mlp_pred', 'gcn_pred', 'gat_pred', 'mag_pred',
                'num_node', 'num_edge', 'num_missing_value_per_node', 'num_tr', 'num_val', 'num_te', 'order_list']


## Set Criterion ------------------------------------------------------------------------------
mae = nn.L1Loss()
rmse = utils.RMSELoss()

### test loss
if not args.only_gnn:
    te_ens_pred = torch.tensor(te_df['lr_pred'])
    te_ens_pred += torch.tensor(te_df['rr_pred'])
    te_ens_pred += torch.tensor(te_df['mlp_pred'])
    te_ens_pred += torch.tensor(te_df['gat_pred'])
    te_ens_pred += torch.tensor(te_df['gcn_pred'])
    te_ens_pred += torch.tensor(te_df['mag_pred'])
    te_ens_pred /= 6.0
else:
    te_ens_pred = torch.tensor(te_df['gat_pred'])
    te_ens_pred += torch.tensor(te_df['gcn_pred'])
    te_ens_pred += torch.tensor(te_df['mag_pred'])
    te_ens_pred /= 3.0

te_df['ens_pred'] = te_ens_pred.detach().numpy()
te_gt = np.array(te_df['te_gt'].tolist())

te_mae_avg_loss = 0; te_rmse_avg_loss = 0
for idx in te_df['graph_idx'].unique():
    idx_ = te_df[te_df['graph_idx'] == idx].index
    out = te_ens_pred[idx_]
    gt = torch.tensor(te_gt[idx_])
    te_mae_loss = mae(out, gt); te_rmse_loss = rmse(out, gt)
    te_mae_avg_loss += te_mae_loss; te_rmse_avg_loss += te_rmse_loss
te_mae_avg_loss /= len(te_df); te_rmse_avg_loss /= len(te_df)

te_mae_group_loss_dict, te_mae_group_loss, te_rmse_group_loss_dict, te_rmse_group_loss = utils.cal_group_loss(te_gt, te_df['ens_pred'])        
te_mae_worst_loss = max(te_mae_group_loss_dict.values())
te_rmse_worst_loss = max(te_rmse_group_loss_dict.values())


if not args.ignore_wandb:
    wandb.log({'test loss (MAE)': te_mae_avg_loss})
    wandb.log({'test loss (RMSE)': te_rmse_avg_loss})
    wandb.log({'test_group_avg_loss (MAE)': te_mae_group_loss})
    wandb.log({'test_group_avg_loss (RMSE)': te_rmse_group_loss})
    wandb.log({'test_worst_loss (MAE)': te_mae_worst_loss})
    wandb.log({'test_worst_loss (RMSE)': te_rmse_worst_loss})
## ----------------------------------------------------------------------------

## ----------------------------------------------------------------------------
## Print
columns = [f"te_data_loss(MAE)",  f"te_group_loss(MAE)", f"te_worst_loss(MAE)"]
values = [te_mae_avg_loss, te_mae_group_loss, te_mae_worst_loss]

table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
print(table)
## ----------------------------------------------------------------------------


    
# %%
## Save results as csv
save_df = pd.DataFrame.from_dict([{"target_order" : args.target_order,
                                    "seed" : args.seed,
                                    "fold" : args.fold,
                                    "test_data_loss (MAE)" : te_mae_avg_loss,
                                    "test_group_loss (MAE)" : te_mae_group_loss,
                                    "test_worst_loss (MAE)" : te_mae_worst_loss,
                                    "test_data_loss (RMSE)" : te_rmse_avg_loss,
                                    "test_group_loss (RMSE)" : te_rmse_group_loss,
                                    "test_worst_loss (RMSE)" : te_rmse_worst_loss,
                                    }])

if not os.path.exists(f'{save_path}/{args.method}_result.csv'):
    save_df.to_csv(f'{save_path}/{args.method}_result.csv', index=False, mode='w', encoding='utf-8-sig')
else:
    save_df.to_csv(f'{save_path}/{args.method}_result.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
