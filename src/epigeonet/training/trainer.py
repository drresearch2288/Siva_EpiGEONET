"""Training loop for EpiGeoNet."""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import mlflow
from pathlib import Path
from loguru import logger
from epigeonet.models.losses import total_loss

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, 
                 lr: float = 1e-3, patience: int = 10, epochs: int = 200,
                 save_dir: str = 'results/models'):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=patience//2)
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def fit(self, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                if 'targets' in batch:
                    for k, v in batch['targets'].items():
                        if isinstance(v, torch.Tensor):
                            batch['targets'][k] = v.to(self.device)
                if 'graph_static' in batch:
                    for k, v in batch['graph_static'].items():
                        if isinstance(v, torch.Tensor):
                            batch['graph_static'][k] = v.to(self.device)
                
                preds, _ = self.model(batch)
                loss, comps = total_loss(preds, batch['targets'], batch['graph_static'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= max(1, len(train_loader))
            
            # Validate
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Log MLflow
            try:
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('val_loss', val_loss, step=epoch)
                mlflow.log_metric('epoch_time', epoch_time, step=epoch)
                if self.device.type == 'mps':
                    mlflow.log_metric('mps_peak_memory_mb', 0.0, step=epoch)
            except Exception:
                pass
            
            logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_dir / 'epigeonet_best.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered.")
                    break
                    
        return best_val_loss
        
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                if 'targets' in batch:
                    for k, v in batch['targets'].items():
                        if isinstance(v, torch.Tensor):
                            batch['targets'][k] = v.to(self.device)
                if 'graph_static' in batch:
                    for k, v in batch['graph_static'].items():
                        if isinstance(v, torch.Tensor):
                            batch['graph_static'][k] = v.to(self.device)
                            
                preds, _ = self.model(batch)
                loss, _ = total_loss(preds, batch['targets'], batch['graph_static'])
                val_loss += loss.item()
        return val_loss / max(1, len(val_loader))
