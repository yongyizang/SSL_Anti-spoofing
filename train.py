import argparse
import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import Model

def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle
    
        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return


def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.asarray(target_scores)
    nontarget_scores = np.asarray(nontarget_scores)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
 
    '''
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    
    parser.add_argument('--phase_perturb_row', action='store_true', default=False,
                        help='use phase perturbation at row-level? (default false)')
    
    parser.add_argument('--phase_perturb_col', action='store_true', default=False,
                        help='use phase perturbation at col-level? (default false)')
    
    parser.add_argument('--rawboost_on', action='store_true', default=False,
                        help='use rawboost? (default false)')
    
    parser.add_argument('--debug', action='store_true', default=False,
                    help='Run in debug mode (1 epoch, 20 steps)')


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    # output dir
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--wandb_project', type=str, default='phase-antispoofing')
    parser.add_argument('--wandb_entity', type=str, default="airlab")
    return parser.parse_args()

def evaluate_model(dev_loader, model, device):
    model.eval()
    val_loss = 0.0
    num_total = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    
    target_scores, nontarget_scores = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader, desc='Validation'):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
            target_scores.extend(batch_out[batch_y == 1, 1].data.cpu().numpy().ravel())
            nontarget_scores.extend(batch_out[batch_y == 0, 1].data.cpu().numpy().ravel())
    
    eer, _ = compute_eer(target_scores, nontarget_scores)
    
    return val_loss / num_total, eer

def train_epoch(train_loader, model, optimizer, device):
    model.train()
    running_loss = 0
    num_total = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    
    target_scores, nontarget_scores = [], []
    for batch_x, batch_y in tqdm(train_loader, desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        optimizer.zero_grad()
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        batch_loss.backward()
        optimizer.step()
        
        running_loss += (batch_loss.item() * batch_size)
        
        target_scores.extend(batch_out[batch_y == 1, 1].data.cpu().numpy().ravel())
        nontarget_scores.extend(batch_out[batch_y == 0, 1].data.cpu().numpy().ravel())
    
    eer, _ = compute_eer(target_scores, nontarget_scores)
    
    return running_loss / num_total, eer

def main():
    args = parse_arguments()
    set_random_seed(args.seed, args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize Wandb only if not in debug mode
    if not args.debug:
        run = wandb.init(project=args.wandb_project, config=args, name=args.comment)
        model_save_path = os.path.join(args.output_dir, run.name)
    else:
        print("Running in debug mode: Wandb logging disabled")
        model_save_path = os.path.join(args.output_dir, 'debug_run')
    
    os.makedirs(model_save_path, exist_ok=True)
    
    # Model initialization
    model = Model(args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data loaders
    train_set, dev_set, eval_set = prepare_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=12, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=12, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=12, shuffle=False)
    
    print(f'Training samples: {len(train_set)}, Validation samples: {len(dev_set)}, Evaluation samples: {len(eval_set)}')
    
    best_val_eer = float('inf')
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    last_model_path = os.path.join(model_save_path, 'last_model.pth')
    
    if args.debug:
        print("Running in debug mode: 1 epoch, 20 steps")
        num_epochs = 1
        debug_train_loader = list(train_loader)[:20]  # Limit to 20 steps
        debug_val_loader = list(dev_loader)[:20]  # Limit to 20 steps
        debug_test_loader = list(eval_loader)[:20]  # Limit to 20 steps
    else:
        num_epochs = args.num_epochs
        debug_train_loader = train_loader
        debug_val_loader = dev_loader
        debug_test_loader = eval_loader
    
    for epoch in range(num_epochs):
        train_loss, train_eer = train_epoch(debug_train_loader, model, optimizer, device)
        val_loss, val_eer = evaluate_model(debug_val_loader, model, device)
        
        if not args.debug:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_eer': train_eer,
                'val_eer': val_eer
            })
        else:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train EER: {train_eer:.4f}, Val EER: {val_eer:.4f}')
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path}')
        
        torch.save(model.state_dict(), last_model_path)
    
    if not args.debug:
        wandb.save(best_model_path)    
        wandb.save(last_model_path)
    
        # Evaluate EER.
        # If in debug mode, we directly eval; if not, we load the best model and eval
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        target_scores, nontarget_scores = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(eval_loader, desc='Evaluation'):
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                target_scores.extend(batch_out[batch_y == 1, 1].data.cpu().numpy().ravel())
                nontarget_scores.extend(batch_out[batch_y == 0, 1].data.cpu().numpy().ravel())
                        
        best_eer, _ = compute_eer(target_scores, nontarget_scores)
        
        # only necessary to load last if not in debug mode
        model.load_state_dict(torch.load(last_model_path))
        
    model.eval()
    
    target_scores, nontarget_scores = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(debug_test_loader, desc='Evaluation'):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            target_scores.extend(batch_out[batch_y == 1, 1].data.cpu().numpy().ravel())
            nontarget_scores.extend(batch_out[batch_y == 0, 1].data.cpu().numpy().ravel())
    
    last_eer, _ = compute_eer(target_scores, nontarget_scores)
    
    if not args.debug:
        wandb.log({
            'best_eer': best_eer,
            'last_eer': last_eer
        })
    
        print(f'Evaluation results - EER: Best: {best_eer:.4f}, Last: {last_eer:.4f}')
    else:
        print(f'Evaluation results - EER: Last: {last_eer:.4f}')
    
    if not args.debug:
        wandb.finish()

def prepare_datasets(args):
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.database_path, f'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True,
        is_eval=False
    )
    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, f'ASVspoof2019_LA_train/'),
        algo=args.algo
    )
    
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.database_path, f'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False,
        is_eval=False
    )
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, f'ASVspoof2019_LA_dev/'),
        algo=args.algo
    )
    
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=os.path.join(args.database_path, 'ASVspoof_LA_cm_protocols/trial_metadata.txt'),
        base_dir=os.path.join(args.database_path, f'ASVspoof2021_LA_eval/')
    )
    
    return train_set, dev_set, eval_set


if __name__ == '__main__':
    main()