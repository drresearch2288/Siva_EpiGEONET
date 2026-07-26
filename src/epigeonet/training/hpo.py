"""Hyperparameter Optimization with Optuna."""
import argparse
import optuna
import yaml
import torch
import mlflow
from pathlib import Path
from epigeonet.models.epigeonet import EpiGeoNet
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

def objective(trial):
    # Hyperparameters to search
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    tcn_channels = trial.suggest_categorical("tcn_channels", [32, 64, 128])
    ascgc_k = trial.suggest_categorical("ascgc_k", [6, 8, 10])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # Loss weights (these would optimally be passed to total_loss, but we just log them for now)
    l1 = trial.suggest_float("l1", 0.1, 1.0)
    l2 = trial.suggest_float("l2", 0.1, 1.0)
    l3 = trial.suggest_float("l3", 0.01, 0.2)
    l4 = trial.suggest_float("l4", 0.01, 0.2)
    
    device = get_device()
    
    # Mock data setup for HPO runs
    ds_train = EpiGeoDataset(split='train')
    ds_val = EpiGeoDataset(split='val')
    
    # If empty (e.g. testing), just create dummy
    if len(ds_train) == 0:
        N = max(ascgc_k + 1, 10)
        sample = {
            'x': torch.randn(12, N, 24),
            'static': torch.randn(N, 1),
            'w_clim': torch.rand(12, N, N),
            'w_case': torch.rand(12, N, N),
            'graph_static': {
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
                'edge_weight': torch.tensor([1.0, 1.0, 1.0]),
                'A_geo': torch.eye(N),
                'w_dist': torch.rand(N, N)
            },
            'targets': {
                'reg': torch.ones(N, 3),
                'risk': torch.zeros(N, dtype=torch.long),
                'alert': torch.zeros(N, 1).float()
            }
        }
        ds_train.samples = [sample, sample]
        ds_val.samples = [sample]
        
    train_loader = DataLoader(ds_train, batch_size=2, collate_fn=epigeonet_collate)
    val_loader = DataLoader(ds_val, batch_size=2, collate_fn=epigeonet_collate)
    
    model = EpiGeoNet(n_nodes=ds_train.samples[0]['x'].shape[1], in_dim=24, explain=False)
    
    trainer = Trainer(model, device=device, lr=lr, epochs=2, patience=1, save_dir='results/models/hpo')
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        val_loss = trainer.fit(train_loader, val_loader)
        mlflow.log_metric('hpo_val_loss', val_loss)
        
    return val_loss

def run_hpo(n_trials: int = 10, study_name: str = "epigeonet-hpo"):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    
    study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    
    # Save best config
    out_dir = Path('config')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'model_best.yaml', 'w') as f:
        yaml.dump(study.best_trial.params, f)
        
    return study

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=10)
    args = parser.parse_args()
    
    run_hpo(n_trials=args.n_trials)

if __name__ == '__main__':
    main()
