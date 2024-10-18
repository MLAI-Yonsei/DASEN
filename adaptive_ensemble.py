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

class LinearRegression(nn.Module):
    def __init__(self, degree):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(degree, 1)
    def forward(self, df):
        x = torch.stack([torch.tensor(df['lr_pred'], dtype=torch.float32, device='cuda'),
                torch.tensor(df['rr_pred'], dtype=torch.float32, device='cuda'),
                torch.tensor(df['mlp_pred'], dtype=torch.float32, device='cuda'),
                torch.tensor(df['gat_pred'], dtype=torch.float32, device='cuda'),
                torch.tensor(df['gcn_pred'], dtype=torch.float32, device='cuda'),
                torch.tensor(df['mag_pred'], dtype=torch.float32, device='cuda')],
                dim=1)
        # x = x.permute(1, 0)
        return self.linear(x)


def svr_loss(output, target, epsilon=0.1):
    loss = torch.max(torch.abs(output - target) - epsilon, torch.tensor(0.0))
    return torch.mean(loss)


## Argparse ------------------------------------------------------
parser = argparse.ArgumentParser(description="")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--target_order", type=int, default=3,
          help="Decide how much order we get (Default : 3)") # 3: 3차까지는 주어진 상태에서, 4차 여부 예측

parser.add_argument("--method", type=str, default='lr',
                    choices=['lr', 'svr'])

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
parser.add_argument("--epochs", type=int, default=200,
        help="number of epochs to optimize the learnable alpha (Default: 200)")

parser.add_argument("--tol", type=int, default=15, metavar="N",
    help="number epochs for early stopping (Default : 15)")

parser.add_argument("--lr_init", type=float, default=5e-3,
                help="learning rate (Default : 0.005)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (Default: 5e-4)")

parser.add_argument("--eps", type=float, default=0.1, help="epsilon for svr loss (Default : 0.1)")

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
    run_name = f"{args.method}-order{args.target_order}-seed{args.seed}-fold{args.fold}-lr{args.lr_init}-wd{args.wd}-epc{args.epochs}-tol{args.tol}"
    if args.rm_feature != -999:
        run_name + f"-rm_feature({args.rm_feature})"
    wandb.run.name = run_name
    
    
## Set alpha or model ---------------------------------------------------------------
model = LinearRegression(degree=6).cuda()
## -----------------------------------------------------------------------------

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
if args.method == 'lr':
    if args.criterion == 'MSE':
        criterion = nn.MSELoss() 
    elif args.criterion == "RMSE":
        criterion = utils.RMSELoss()
elif args.method == 'svr':
    criterion = svr_loss
        
if args.eval_criterion == 'MAE':
    eval_criterion = nn.L1Loss()
elif args.eval_criterion == "RMSE":
    eval_criterion = utils.RMSELoss()
mae = nn.L1Loss()
rmse = utils.RMSELoss()


## Set Optimizer and Scheduler ------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)


## Training
columns = ["ep", "lr", f"tr_loss({args.criterion})", 
        f"val_data_loss({args.eval_criterion})", f"val_group_loss({args.eval_criterion})", f"val_worst_loss({args.eval_criterion})",
        f"te_data_loss(MAE)",  f"te_group_loss(MAE)", f"te_worst_loss(MAE)",
        "time"]

best_val_group_loss = 99999999 ; cnt = 0
for epoch in range(1, args.epochs + 1):
    time_ep = time.time()
    lr = optimizer.param_groups[0]['lr']
    
    ##-------------------------------------------------------------------------
    ### Forward
    tr_ens_pred = model(tr_df)
    tr_df['ens_pred'] = tr_ens_pred.detach().cpu().numpy()
    tr_gt = torch.tensor(tr_df['tr_gt'], dtype=torch.float32, device='cuda')
    
    ### training loss
    tr_avg_loss = 0
    for idx in tr_df['graph_idx'].unique():
        idx_ = tr_df[tr_df['graph_idx'] == idx].index
        out = tr_ens_pred[idx_]
        # gt = torch.tensor(tr_gt[idx_])
        gt = tr_gt[idx_]
        if args.method == 'lr':
            tr_loss = criterion(out, gt)
        elif args.method == 'svr':
            tr_loss = criterion(out, gt, epsilon=args.eps)
        tr_avg_loss += tr_loss
    tr_avg_loss /= len(tr_df)
        
    if not args.ignore_wandb:
        wandb.log({'train_loss': tr_avg_loss})
    ##-------------------------------------------------------------------------
    
    ##-------------------------------------------------------------------------
    ### validation loss
    val_ens_pred = model(val_df)
    val_df['ens_pred'] = val_ens_pred.detach().cpu().numpy()
    val_gt = torch.tensor(val_df['val_gt'], dtype=torch.float32, device='cuda')
    
    val_avg_loss = 0
    for idx in val_df['graph_idx'].unique():
        idx_ = val_df[val_df['graph_idx'] == idx].index
        out = val_ens_pred[idx_]
        gt = val_gt[idx_]
        val_loss = eval_criterion(out, gt)
        val_avg_loss += val_loss
    val_avg_loss /= len(val_df)            

    val_group_loss_dict, val_group_loss, _, _ = utils.cal_group_loss(val_gt.detach().cpu().numpy(), val_df['ens_pred'])        
    val_worst_loss = max(val_group_loss_dict.values())
    
    if not args.ignore_wandb:
        wandb.log({'valid_data_loss (MSE)': val_avg_loss})
        wandb.log({'valid_group_avg_loss (MAE)': val_group_loss})
        wandb.log({'valid_worst_loss (MAE)': val_worst_loss})

    ## ----------------------------------------------------------------------------
    ### test loss
    te_ens_pred = model(te_df)
    te_df['ens_pred'] = te_ens_pred.detach().cpu().numpy()
    te_gt = torch.tensor(te_df['te_gt'], dtype=torch.float32, device='cuda')
    
    te_mae_avg_loss = 0; te_rmse_avg_loss = 0
    for idx in te_df['graph_idx'].unique():
        idx_ = te_df[te_df['graph_idx'] == idx].index
        out = te_ens_pred[idx_]
        gt = te_gt[idx_]
        te_mae_loss = mae(out, gt); te_rmse_loss = rmse(out, gt)
        te_mae_avg_loss += te_mae_loss; te_rmse_avg_loss += te_rmse_loss
    te_mae_avg_loss /= len(te_df); te_rmse_avg_loss /= len(te_df)

    te_mae_group_loss_dict, te_mae_group_loss, te_rmse_group_loss_dict, te_rmse_group_loss = utils.cal_group_loss(te_gt.detach().cpu().numpy(), te_df['ens_pred'])        
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
    time_ep = time.time() - time_ep

    values = [epoch, lr, tr_avg_loss,
            val_avg_loss, val_group_loss, val_worst_loss,
            te_mae_avg_loss, te_mae_group_loss, te_mae_worst_loss,
            time_ep,]

    table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0 or epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    ## ----------------------------------------------------------------------------
    
    ## ----------------------------------------------------------------------------
    ## Save Best Model (Early Stopping)
    os.makedirs(save_path, exist_ok=True)
    # based on group loss
    if val_group_loss < best_val_group_loss:
        cnt = 0
        best_epoch = epoch
        best_w = model.state_dict()
            
        best_val_data_loss = val_avg_loss.item()
        best_val_group_loss = val_group_loss
        best_val_worst_loss = val_worst_loss

        best_te_mae_data_loss = te_mae_avg_loss.item()
        best_te_rmse_data_loss = te_rmse_avg_loss.item()
        best_te_mae_group_loss = te_mae_group_loss
        best_te_rmse_group_loss = te_rmse_group_loss
        best_te_mae_worst_loss = te_mae_worst_loss
        best_te_rmse_worst_loss = te_rmse_worst_loss

        # save state_dict
        utils.save_checkpoint(file_path = f"{save_path}/{args.method}_best_val_cv_tr{args.fold}.pt",
                            epoch = epoch,
                            w = best_w,
                            optimizer = optimizer.state_dict(),
                            scheduler = scheduler.state_dict()
                            )
        
        # save prediction of best model and ground truth as csv
        tr_ens_df = pd.DataFrame({'tr_pred':tr_df['ens_pred'],
                        'tr_ground_truth':tr_gt.detach().cpu().numpy(),
                        'num_node': tr_df['num_node'],
                        'num_edge': tr_df['num_edge'],
                        'num_missing_value_per_node': tr_df['num_missing_value_per_node'],
                        'num_tr': tr_df['num_tr'],
                        'num_val': tr_df['num_val'],
                        'num_te': tr_df['num_te'],
                        'order_list': tr_df['order_list']})
        tr_ens_df.to_csv(f"{save_path}/{args.method}_best_tr_pred_cv_tr{args.fold}.csv")
        val_ens_df = pd.DataFrame({'val_pred':val_df['ens_pred'],
                            'val_ground_truth':val_gt.detach().cpu().numpy(),
                            'num_node': val_df['num_node'],
                            'num_edge': val_df['num_edge'],
                            'num_missing_value_per_node': val_df['num_missing_value_per_node'],
                            'num_tr': val_df['num_tr'],
                            'num_val': val_df['num_val'],
                            'num_te': val_df['num_te'],
                            'order_list': val_df['order_list']})
        val_ens_df.to_csv(f"{save_path}/{args.method}_best_val_pred_cv_tr{args.fold}.csv")
        te_ens_df = pd.DataFrame({'te_pred' : te_df['ens_pred'],
                            'te_ground_truth' : te_gt.detach().cpu().numpy(),
                            'graph_index': te_df['graph_idx'],
                            'num_node': te_df['num_node'],
                            'num_edge': te_df['num_edge'],
                            'num_missing_value_per_node': te_df['num_missing_value_per_node'],
                            'num_tr': te_df['num_tr'],
                            'num_val': te_df['num_val'],
                            'num_te': te_df['num_te'],
                            'order_list': te_df['order_list']})             
        te_ens_df.to_csv(f"{save_path}/{args.method}_best_te_pred_cv_tr{args.fold}.csv")
    else:
        cnt += 1
        
    if cnt == args.tol or epoch==args.epochs:
        break

    tr_avg_loss.backward()
    optimizer.step()        
    scheduler.step()
    optimizer.zero_grad()
    ## ----------------------------------------------------------------------------
    
## Save results as csv
save_df = pd.DataFrame.from_dict([{"target_order" : args.target_order,
                                "seed" : args.seed,
                                "fold" : args.fold,
                                "epoch" : args.epochs,
                                "best_epoch" : best_epoch,
                                "lr_init" : args.lr_init,
                                "wd" : args.wd,
                                "val_data_loss (MAE)" : best_val_data_loss,
                                "val_group_loss (MAE)" : best_val_group_loss,
                                "val_worst_loss (MAE)" : best_val_worst_loss,
                                "test_data_loss (MAE)" : best_te_mae_data_loss,
                                "test_group_loss (MAE)" : best_te_mae_group_loss,
                                "test_worst_loss (MAE)" : best_te_mae_worst_loss,
                                "test_data_loss (RMSE)" : best_te_rmse_data_loss,
                                "test_group_loss (RMSE)" : best_te_rmse_group_loss,
                                "test_worst_loss (RMSE)" : best_te_rmse_worst_loss,
                                }])


if not os.path.exists(f'{save_path}/{args.method}_result.csv'):
    save_df.to_csv(f'{save_path}/{args.method}_result.csv', index=False, mode='w', encoding='utf-8-sig')
else:
    save_df.to_csv(f'{save_path}/{args.method}_result.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
