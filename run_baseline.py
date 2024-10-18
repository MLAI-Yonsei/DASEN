import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import os, time

import argparse
import tabulate

import utils, models
import wandb


import warnings
warnings.filterwarnings('ignore')

## Argparse ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Graphical Medical-AI")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

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

parser.add_argument(
    "--birthyear_rule",
    type=str,
    default="mdlp",
    help="birthyear_chunk_id: division with 30 years interval, mdlp: Minimum Description Length Binning"
)

parser.add_argument(
    "--replace_missing_value",
    type=str,
    default='replace',
    choices=['replace', 'mice', 'binomial']
)

parser.add_argument("--num_folds", type=int, default=3,  
            help="Number of folds when running cross validation (Defulat : 3)")

parser.add_argument("--avg_cv_results", action="store_true", default=True,
            help="Averaging the result of each fold")   
#----------------------------------------------------------------


# Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='MagNet',
    choices=["MagNet", "GCN", "GAT", "MLP", "Linear", "Ridge"],
    help="model name (default : MagNet)")

parser.add_argument("--save_path",
            type=str, default="./exp_result/",
            help="Path to save best model dict")

parser.add_argument(
    "--num_features",
    type=int, default=17,
    help="feature size (default : 17)"
)

parser.add_argument(
    "--q",
    type=float, default=0.25,
    help="Directional flow (default : 0.25)"
)

parser.add_argument(
    "--K",          
    type=int, default=2,
    help=""
)

parser.add_argument(
    "--hidden_dim",
    type=int, default=16,
    help="MagNet model hidden size (default : 16)"
)

parser.add_argument(
    "--num_layers",
    type=int, default=2,
    help="number of layers (default : 2)"
)

parser.add_argument(
    "--drop_out",
    type=float, default=0.0,
    help="Dropout Rate (Default : 0)"
)

parser.add_argument(
    "--complex_activation",
    action='store_true',
    help='Use complex activation for MagNet (Default : False)'
)

parser.add_argument(
    "--trainable_q",
    action='store_true',
    help='Set trainable q for MagNet (Default : False)'
)

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

# Learning Hyperparameter --------------------------------------
parser.add_argument("--lr_init", type=float, default=0.005,
                help="learning rate (Default : 0.005)")

parser.add_argument("--optim", type=str, default="adam",
                    choices=["sgd", "adam"],
                    help="Optimization options")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=100, metavar="N",
    help="number epochs to train (Default : 100)")

parser.add_argument("--tol", type=int, default=20, metavar="N",
    help="number epochs for early stopping (Default : 20)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (Default: 5e-4)")

parser.add_argument("--scheduler", type=str, default="cos_anneal", choices=[None, "cos_anneal"])

parser.add_argument("--lamb", type=float, default=1.0,
                help="Penalty term for Ridge Regression (Default : 1)")
#----------------------------------------------------------------

# Hyperparameter for setting --------------------------------------
parser.add_argument("--target_order", type=int, default=3,
          help="Decide how much order we get (Default : 3)") # 3: 3차까지는 주어진 상태에서, 4차 여부 예측

# parser.add_argument("--train_ratio", type=float, default=0.2,
#           help="Ratio of train data (Default : 0.2)") # 0.8: n차 감염의 20%를 train으로 활용
#----------------------------------------------------------------

args = parser.parse_args()
## ----------------------------------------------------------------------------------------------------



## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

save_path = args.save_path + f'exp_result/{args.target_order}-{args.seed}/{args.model}-{args.optim}'
if args.rm_feature != -999:
    save_path = save_path + f'/rm{args.rm_feature}'

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
birthyear = 'mdlp' if args.birthyear_rule == 'mdlp' else 'heuristic'
replace = args.replace_missing_value
if args.ignore_wandb == False:
    wandb.init(project=args.wandb_name, entity=args.wandb_entity)
    wandb.config.update(args)

    if args.model not in ['MagNet']:
        run_name = f"{birthyear}-{replace}-cv-{args.target_order}-{args.seed}-{args.model}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.epochs}"
    else:
        run_name = f"{birthyear}-{replace}-cv-{args.target_order}-{args.seed}-{args.model}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.epochs}-{args.q}-{args.K}"
    
    if args.rm_feature != -999:
        run_name = run_name + f"-rm_feature({args.rm_feature})"

    wandb.run.name = run_name

cv_val_data_loss = []; cv_val_group_loss = []; cv_val_worst_loss = []            
cv_te_mae_data_loss = []; cv_te_rmse_data_loss = []
cv_te_mae_group_loss = []; cv_te_rmse_group_loss = []
cv_te_mae_worst_loss = []; cv_te_rmse_worst_loss = []
for tr_fold in range(args.num_folds):
    ## Load Data --------------------------------------------------------------------------------
    final_data_list, y_min, y_max, num_birthyear_disc = utils.full_load_data_cv(data_path = args.data_path,  
                                                                    num_features = args.num_features,
                                                                    target_order = args.target_order,
                                                                    num_folds= args.num_folds,
                                                                    tr_fold = tr_fold,
                                                                    classification = True,
                                                                    device = args.device,
                                                                    model_name = args.model,
                                                                    args = args)
    print(f"Successfully load data with CV (Training : {tr_fold+1}/{args.num_folds})!")
    #-------------------------------------------------------------------------------------

    ## Model ------------------------------------------------------------------------------------
    if args.model == "MagNet":
        model_type = 'graph'
        model = models.MagNet(hidden=args.hidden_dim,
                            q=args.q,
                            K=args.K,
                            drop_out=args.drop_out, 
                            activation=args.complex_activation,
                            trainable_q=args.trainable_q,
                            num_birthyear_disc=num_birthyear_disc,
                            replace_missing_value=args.replace_missing_value,
                            layer=args.num_layers,
                            rm_feature=args.rm_feature).to(args.device) 

    elif args.model == "GCN":
        model_type = 'graph'
        model = models.GCN_Net(hidden_dim=args.hidden_dim,
                            drop_out=args.drop_out, 
                            num_birthyear_disc=num_birthyear_disc,
                            replace_missing_value=args.replace_missing_value,
                            rm_feature=args.rm_feature).to(args.device)

    elif args.model == "GAT":
        model_type = 'graph'
        model = models.GAT_Net(hidden_dim=args.hidden_dim,
                            drop_out=args.drop_out, 
                            num_birthyear_disc=num_birthyear_disc,
                            replace_missing_value=args.replace_missing_value,
                            heads=args.num_layers,
                            rm_feature=args.rm_feature).to(args.device)

    elif args.model == "MLP":
        model_type = 'non_graph'
        model = models.MLPRegressor(hidden_size=args.hidden_dim,
                                drop_out=args.drop_out,
                                num_birthyear_disc=num_birthyear_disc,
                                replace_missing_value=args.replace_missing_value,
                                rm_feature=args.rm_feature).to(args.device)

    elif args.model == "Linear":
        model_type = 'non_graph'
        model = models.LinearRegression(out_channels=1,
                        num_birthyear_disc=num_birthyear_disc, rm_feature=args.rm_feature).to(args.device)

    elif args.model == "Ridge":
        model_type = "ridge"
        model = models.LinearRegression(out_channels=1,
                        num_birthyear_disc=num_birthyear_disc, rm_feature=args.rm_feature).to(args.device)

    print(f"Successfully prepare {args.model} model")
    
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

        print(f"Successfully load trained {args.model} model from {args.resume}")
    # ---------------------------------------------------------------------------------------------


    ## Criterion ------------------------------------------------------------------------------
    # Train Criterion
    if args.criterion == 'MSE':
        criterion = nn.MSELoss() 

    elif args.criterion == "RMSE":
        criterion = utils.RMSELoss()

    # Validation / Test Criterion
    if args.eval_criterion == 'MAE':
        eval_criterion = nn.L1Loss()
    elif args.eval_criterion == "RMSE":
        eval_criterion = utils.RMSELoss()

    mae = nn.L1Loss()
    rmse = utils.RMSELoss()
    # ---------------------------------------------------------------------------------------------

    ## Optimizer and Scheduler --------------------------------------------------------------------
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    else:
        raise NotImplementedError

    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.scheduler  == "cos_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)    
    else:
        scheduler = None

    if args.resume is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    # ---------------------------------------------------------------------------------------------


    ## Training Phase -----------------------------------------------------------------------------
    columns = ["ep", "lr", f"tr_loss({args.criterion})", 
            f"val_data_loss({args.eval_criterion})", f"val_group_loss({args.eval_criterion})", f"val_worst_loss({args.eval_criterion})",
            f"te_data_loss(MAE)",  f"te_group_loss(MAE)", f"te_worst_loss(MAE)",
            "time"]

    best_val_group_loss = 99999999 ; cnt = 0
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()
        lr = optimizer.param_groups[0]['lr']

        tr_epoch_loss = 0; val_epoch_loss = 0; te_mae_epoch_loss = 0 ; te_rmse_epoch_loss = 0

        total_tr_num_data = 0; total_val_num_data = 0; total_te_num_data = 0

        tr_ground_truth_list = []; val_ground_truth_list = []; te_ground_truth_list = []

        tr_predicted_list = []; val_predicted_list = []; te_predicted_list = []

        total_loss = 0
        for itr, data in enumerate(final_data_list):
            # tr_loss, tr_num_data, tr_predicted, tr_ground_truth = utils.train(data, model, model_type, optimizer, criterion)
            model.train()
            data_x, data_edge, data_y, data_tr_mask, _, _, _ = data
            num_data = torch.sum(data_tr_mask)

            optimizer.zero_grad()

            if model_type in ['ridge', 'non_graph']:
                out = model(data_x).squeeze()
            elif model_type == 'graph':
                out = model(x=data_x, edge_index=data_edge).squeeze()

            loss = criterion(out[data_tr_mask], data_y[data_tr_mask])
            if torch.isnan(loss):
                continue
            if model_type == 'ridge':
                loss += args.lamb*torch.norm(model.linear1.weight, p=2)
            tr_loss = loss.item()
            total_loss += loss / num_data
            tr_num_data = num_data
            tr_predicted = torch.clamp(out[data_tr_mask], min=y_min, max=y_max)
            tr_ground_truth = data_y[data_tr_mask]
            tr_epoch_loss += tr_loss
            total_tr_num_data += tr_num_data
            tr_ground_truth_list += list(tr_ground_truth.cpu().numpy())
            tr_predicted_list += list(tr_predicted.detach().cpu().numpy())

        total_loss.backward()
        optimizer.step()

        # for qualitative analysis
        tr_graph_index_list = []
        val_graph_index_list = []
        te_graph_index_list = []
        
        tr_num_node_list = []
        val_num_node_list = []
        te_num_node_list = []

        tr_num_edge_list = []
        val_num_edge_list = []
        te_num_edge_list = []

        tr_num_missing_value_list = []
        val_num_missing_value_list = []
        te_num_missing_value_list = []

        tr_num_tr_list = []
        val_num_tr_list = []
        te_num_tr_list = []

        tr_num_val_list = []
        val_num_val_list = []
        te_num_val_list = []

        tr_num_te_list = []
        val_num_te_list = []
        te_num_te_list = []

        tr_order_list = []
        val_order_list = []
        te_order_list = []
        for itr, data in enumerate(final_data_list):
            # trainset data analysis
            data_x, data_edge, data_y, data_tr_mask, data_val_mask, data_te_mask, _ = data
            tr_num_missing_value = torch.sum(data_x[data_tr_mask] == 2, axis=1).tolist()
            val_loss, val_num_data, val_predicted, val_ground_truth, val_num_missing_value = utils.valid(data, model, model_type, eval_criterion, y_min, y_max, args.lamb)

            te_mae_loss, te_rmse_loss, te_num_data, te_predicted, te_ground_truth, te_num_missing_value, num_tr, num_val, num_te \
             = utils.test(data, model, model_type, mae, rmse, y_min, y_max, args.lamb)
            
            val_epoch_loss += val_loss
            te_mae_epoch_loss += te_mae_loss
            te_rmse_epoch_loss += te_rmse_loss
            total_val_num_data += val_num_data
            total_te_num_data += te_num_data

            # qualitative anaylsis - train set
            tr_graph_index_list += [itr] * num_tr
            tr_num_node_list += [len(data[0])] * num_tr
            tr_num_edge_list += [len(data[0]) - 1] * num_tr # edge = node - 1
            tr_num_missing_value_list += tr_num_missing_value
            tr_num_tr_list += [num_tr] * num_tr
            tr_num_val_list += [num_val] * num_tr
            tr_num_te_list += [num_te] * num_tr
            tr_order_list += [args.target_order] * num_tr
            
            # qualitative analysis - valid set
            val_graph_index_list += [itr] * num_val
            val_num_node_list += [len(data[0])] * num_val
            val_num_edge_list += [len(data[0]) - 1] * num_val # edge = node - 1
            val_num_missing_value_list += val_num_missing_value
            val_num_tr_list += [num_tr] * num_val
            val_num_val_list += [num_val] * num_val
            val_num_te_list += [num_te] * num_val
            val_order_list += [args.target_order] * num_val

            # qualitative anaylsis - test set
            te_graph_index_list += [itr] * num_te
            te_num_node_list += [len(data[0])] * num_te
            te_num_edge_list += [len(data[0]) - 1] * num_te # edge = node - 1
            te_num_missing_value_list += te_num_missing_value
            te_num_tr_list += [num_tr] * num_te
            te_num_val_list += [num_val] * num_te
            te_num_te_list += [num_te] * num_te
            te_order_list += [args.target_order] * num_te

            val_ground_truth_list += list(val_ground_truth.cpu().numpy())
            te_ground_truth_list += list(te_ground_truth.cpu().numpy())
            val_predicted_list += list(val_predicted.detach().cpu().numpy())
            te_predicted_list += list(te_predicted.detach().cpu().numpy())
        # Data Loss
        tr_avg_loss = tr_epoch_loss / total_tr_num_data
        val_avg_loss = val_epoch_loss / total_val_num_data
        te_mae_avg_loss = te_mae_epoch_loss / total_te_num_data
        te_rmse_avg_loss = te_rmse_epoch_loss / total_te_num_data
        
        tr_ground_truth_list = np.asarray(tr_ground_truth_list)
        val_ground_truth_list = np.asarray(val_ground_truth_list)
        te_ground_truth_list = np.asarray(te_ground_truth_list)

        tr_predicted_list = np.asarray(tr_predicted_list)
        val_predicted_list = np.asarray(val_predicted_list)
        te_predicted_list = np.asarray(te_predicted_list)
        

        # Calculate Validation Group loss
        val_group_loss_dict, val_group_loss, _, _ = utils.cal_group_loss(val_ground_truth_list, val_predicted_list)
        val_worst_loss = max(val_group_loss_dict.values())

        # Calculate Test Group loss
        te_mae_group_loss_dict, te_mae_group_loss, te_rmse_group_loss_dict, te_rmse_group_loss = utils.cal_group_loss(te_ground_truth_list, te_predicted_list)
        te_mae_worst_loss = max(te_mae_group_loss_dict.values())
        te_rmse_worst_loss = max(te_rmse_group_loss_dict.values())

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

        if args.scheduler is not None:
            scheduler.step()

        ## Save Best Model (Early Stopping)
        os.makedirs(save_path, exist_ok=True)

        # based on group loss
        if val_group_loss < best_val_group_loss:
            cnt = 0
            best_epoch_2 = epoch

            best_val_data_loss_2 = val_avg_loss
            best_val_group_loss_2 = val_group_loss
            best_val_worst_loss_2 = val_worst_loss

            best_te_mae_data_loss_2 = te_mae_avg_loss
            best_te_rmse_data_loss_2 = te_rmse_avg_loss
            best_te_mae_group_loss_2 = te_mae_group_loss
            best_te_rmse_group_loss_2 = te_rmse_group_loss
            best_te_mae_worst_loss_2 = te_mae_worst_loss
            best_te_rmse_worst_loss_2 = te_rmse_worst_loss

            best_val_group_loss = val_group_loss

            # save state_dict
            utils.save_checkpoint(file_path = f"{save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_val_cv_tr{tr_fold+1}.pt",
                                epoch = epoch,
                                state_dict = model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                scheduler = scheduler.state_dict()
                                )
            
            # save prediction of best model and ground truth as csv\
            tr_df = pd.DataFrame({'tr_pred':tr_predicted_list,
                                  'tr_ground_truth':tr_ground_truth_list,
                                  'graph_index': tr_graph_index_list,
                                  'num_node': tr_num_node_list,
                                  'num_edge': tr_num_edge_list,
                                  'num_missing_value_per_node': tr_num_missing_value_list,
                                  'num_tr': tr_num_tr_list,
                                  'num_val': tr_num_val_list,
                                  'num_te': tr_num_te_list,
                                  'order_list': tr_order_list})
            tr_df.to_csv(f"{save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_tr_pred_cv_tr{tr_fold+1}.csv")
            val_df = pd.DataFrame({'val_pred':val_predicted_list,
                                  'val_ground_truth':val_ground_truth_list,
                                  'graph_index': val_graph_index_list,
                                  'num_node': val_num_node_list,
                                  'num_edge': val_num_edge_list,
                                  'num_missing_value_per_node': val_num_missing_value_list,
                                  'num_tr': val_num_tr_list,
                                  'num_val': val_num_val_list,
                                  'num_te': val_num_te_list,
                                  'order_list': val_order_list})
            val_df.to_csv(f"{save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_val_pred_cv_tr{tr_fold+1}.csv")
            te_df = pd.DataFrame({'te_pred' : te_predicted_list,
                                  'te_ground_truth' : te_ground_truth_list,
                                  'graph_index': te_graph_index_list,
                                  'num_node': te_num_node_list,
                                  'num_edge': te_num_edge_list,
                                  'num_missing_value_per_node': te_num_missing_value_list,
                                  'num_tr': te_num_tr_list,
                                  'num_val': te_num_val_list,
                                  'num_te': te_num_te_list,
                                  'order_list': te_order_list})                
            te_df.to_csv(f"{save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_te_pred_cv_tr{tr_fold+1}.csv")
        
        else:
            cnt += 1
        
        if args.ignore_wandb == False:
            wandb.log({"lr" : lr,
                    f"Training loss_cv{tr_fold+1}" : tr_avg_loss,
                    f"Validation data loss_cv{tr_fold+1}": val_avg_loss, f"Validation group loss_cv{tr_fold+1}": val_group_loss, f"Validation worst loss_cv{tr_fold+1}" : val_worst_loss, 
                    f"Test data loss (MAE)_cv{tr_fold+1}" : te_mae_avg_loss, f"Test group loss (MAE)_cv{tr_fold+1}" : te_mae_group_loss, f"Test worst loss (MAE)_cv{tr_fold+1}" : te_mae_worst_loss,
                    f"Test data loss (RMSE)_cv{tr_fold+1}" : te_rmse_avg_loss, f"Test group loss (RMSE)_cv{tr_fold+1}" : te_rmse_group_loss, f"Test worst loss (RMSE)_cv{tr_fold+1}" : te_rmse_worst_loss,
                    })
        
            wandb.run.summary[f"best_val_data_loss_cv{tr_fold+1}"] = best_val_data_loss_2
            wandb.run.summary[f"best_val_group_loss_cv{tr_fold+1}"] = best_val_group_loss_2
            wandb.run.summary[f"best_val_worst_loss_cv{tr_fold+1}"] = best_val_worst_loss_2

            wandb.run.summary[f"best_test_loss_cv{tr_fold+1}(MAE)"] = best_te_mae_data_loss_2
            wandb.run.summary[f"best_test_group_loss_cv{tr_fold+1}(MAE)"] = best_te_mae_group_loss_2
            wandb.run.summary[f"best_test_worst_loss_cv{tr_fold+1}(MAE)"] = best_te_mae_worst_loss_2

            wandb.run.summary[f"best_test_loss_cv{tr_fold+1}(RMSE)"] = best_te_rmse_data_loss_2
            wandb.run.summary[f"best_test_group_loss_cv{tr_fold+1}(RMSE)"] = best_te_rmse_group_loss_2
            wandb.run.summary[f"best_test_worst_loss_cv{tr_fold+1}(RMSE)"] = best_te_rmse_worst_loss_2
        
        if (cnt == args.tol) or (epoch == args.epochs):
            cv_val_data_loss.append(best_val_data_loss_2)
            cv_val_group_loss.append(best_val_group_loss_2)
            cv_val_worst_loss.append(best_val_worst_loss_2)
            
            cv_te_mae_data_loss.append(best_te_mae_data_loss_2)
            cv_te_rmse_data_loss.append(best_te_rmse_data_loss_2)
            cv_te_mae_group_loss.append(best_te_mae_group_loss_2)
            cv_te_rmse_group_loss.append(best_te_rmse_group_loss_2)
            cv_te_mae_worst_loss.append(best_te_mae_worst_loss_2)
            cv_te_rmse_worst_loss.append(best_te_rmse_worst_loss_2)
        
        if cnt == args.tol:
            break
        
    # if best_cv_loss > best_val_group_loss:
    #     best_cv = tr_fold +1
    #     best_cv_loss = best_val_group_loss

    #     best_cv_epoch = best_epoch_2

    #     best_cv_val_data_loss = best_val_data_loss_2
    #     best_cv_val_group_loss = best_val_group_loss_2
    #     best_cv_val_wrost_loss = best_val_worst_loss_2

    #     best_cv_te_mae_data_loss = best_te_mae_data_loss_2
    #     best_cv_te_rmse_data_loss = best_te_rmse_data_loss_2
    #     best_cv_te_mae_group_loss = best_te_mae_group_loss_2
    #     best_cv_te_rmse_group_loss = best_te_rmse_group_loss_2
    #     best_cv_te_mae_worst_loss = best_te_mae_worst_loss_2
    #     best_cv_te_rmse_wrost_loss = best_te_rmse_worst_loss_2


## Calculate CV loss
cv_val_data_loss = sum(cv_val_data_loss) / len(cv_val_data_loss)
cv_val_group_loss = sum(cv_val_group_loss)/len(cv_val_group_loss)
cv_val_worst_loss = sum(cv_val_worst_loss) / len(cv_val_worst_loss)            
cv_te_mae_data_loss = sum(cv_te_mae_data_loss) / len(cv_te_mae_data_loss); cv_te_rmse_data_loss = sum(cv_te_rmse_data_loss) / len(cv_te_rmse_data_loss)
cv_te_mae_group_loss = sum(cv_te_mae_group_loss) / len(cv_te_mae_group_loss); cv_te_rmse_group_loss = sum(cv_te_rmse_group_loss) / len(cv_te_rmse_group_loss)
cv_te_mae_worst_loss = sum(cv_te_mae_worst_loss) / len(cv_te_mae_worst_loss); cv_te_rmse_worst_loss = sum(cv_te_rmse_worst_loss) / len(cv_te_rmse_worst_loss)
# ---------------------------------------------------------------------------------------------



## Print Best Model ---------------------------------------------------------------------------
print(f"Best {args.model} achieved {cv_te_mae_group_loss:8.4f}(MAE) with {args.num_folds}-CV")
if args.ignore_wandb == False:
    # Based on Group
    # wandb.run.summary["best_epoch_2"]  = best_epoch_2
    wandb.run.summary["best_val_data_loss_2"] = cv_val_data_loss
    wandb.run.summary["best_val_group_loss_2"] = cv_val_group_loss
    wandb.run.summary["best_val_worst_loss_2"] = cv_val_worst_loss

    wandb.run.summary["best_test_loss_2(MAE)"] = cv_te_mae_data_loss
    wandb.run.summary["best_test_group_loss_2(MAE)"] = cv_te_mae_group_loss
    wandb.run.summary["best_test_worst_loss_2(MAE)"] = cv_te_mae_worst_loss

    wandb.run.summary["best_test_loss_2(RMSE)"] = cv_te_rmse_data_loss
    wandb.run.summary["best_test_group_loss_2(RMSE)"] = cv_te_rmse_group_loss
    wandb.run.summary["best_test_worst_loss_2(RMSE)"] = cv_te_rmse_worst_loss
    
    # Numer of Data
    # wandb.run.summary["Number of Traing Data"] = total_tr_num_data
    # wandb.run.summary["Number of Validation Data"] : total_val_num_data
    # wandb.run.summary["Number of Test Data"] : total_te_num_data
# ---------------------------------------------------------------------------------------------