#!/usr/bin/env python3
"""
Task1: Flower Classification on Oxford 102 Flowers
- Dataset: manually downloaded jpg folder + imagelabels.mat + setid.mat
- Models: ResNet18 (pretrained/scratch), Swin-Tiny (pretrained)
- Experiments: hyperparameter tuning, ablation, ViT comparison
- Logging: wandb / swanlab
"""

import os
import csv
import random
import copy
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.io import loadmat

# Optional logging
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
try:
    import swanlab
    swanlab_available = True
except ImportError:
    swanlab_available = False


# ----------------------------------------------------------------------
# 1. Custom Dataset for Oxford 102 with .mat files
# ----------------------------------------------------------------------
class Oxford102Dataset(Dataset):
    """Oxford 102 Flowers dataset from manually downloaded files."""
    def __init__(self, image_dir, label_mat_path, setid_mat_path, split='train', transform=None):
        """
        Args:
            image_dir: directory containing all image_XXXX.jpg files
            label_mat_path: path to imagelabels.mat
            setid_mat_path: path to setid.mat
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load .mat files
        labels_mat = loadmat(label_mat_path)
        setid_mat = loadmat(setid_mat_path)

        # labels: shape (1, N) -> flatten to (N,), convert to 0-index
        self.labels = labels_mat['labels'][0] - 1

        # Get indices for split (1-indexed in mat file)
        if split == 'train':
            indices = setid_mat['trnid'][0] - 1
        elif split == 'val':
            indices = setid_mat['valid'][0] - 1
        elif split == 'test':
            indices = setid_mat['tstid'][0] - 1
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")

        self.indices = indices
        self.num_classes = 102

        # Build image paths
        self.image_paths = []
        for idx in self.indices:
            img_id = idx + 1  # back to 1-indexed for filename
            fname = f"image_{img_id:05d}.jpg"
            self.image_paths.append(os.path.join(image_dir, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------
# 2. DataModule: handles transforms, data loading, splits
# ----------------------------------------------------------------------
class FlowerDataModule:
    def __init__(self, data_root, batch_size=32, num_workers=4, seed=42):
        """
        Args:
            data_root: directory containing 'jpg' (or images directly) and .mat files
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Locate image folder
        self.image_dir = os.path.join(data_root, 'jpg')
        if not os.path.exists(self.image_dir):
            self.image_dir = data_root  # assume images are directly in data_root

        self.label_mat = os.path.join(data_root, 'imagelabels.mat')
        self.setid_mat = os.path.join(data_root, 'setid.mat')

        # Validate
        assert os.path.exists(self.image_dir), f"Image dir not found: {self.image_dir}"
        assert os.path.exists(self.label_mat), f"Label file not found: {self.label_mat}"
        assert os.path.exists(self.setid_mat), f"Setid file not found: {self.setid_mat}"

        self._prepare_datasets()
        self.num_classes = 102

    def _prepare_datasets(self):
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Validation / test transforms
        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = Oxford102Dataset(
            self.image_dir, self.label_mat, self.setid_mat,
            split='train', transform=train_transform
        )
        self.val_dataset = Oxford102Dataset(
            self.image_dir, self.label_mat, self.setid_mat,
            split='val', transform=val_test_transform
        )
        self.test_dataset = Oxford102Dataset(
            self.image_dir, self.label_mat, self.setid_mat,
            split='test', transform=val_test_transform
        )

    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# 3. Model Factory: create ResNet (pretrained/scratch) or Swin-T
# ----------------------------------------------------------------------
class ModelFactory:
    @staticmethod
    def create_resnet(model_name='resnet18', num_classes=102, pretrained=True):
        if model_name == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
        elif model_name == 'resnet34':
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet: {model_name}")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def create_swin_tiny(num_classes=102, pretrained=True):
        if pretrained:
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        else:
            model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def create_model(model_type, num_classes, pretrained=True):
        if model_type.startswith('resnet'):
            return ModelFactory.create_resnet(model_type, num_classes, pretrained)
        elif model_type == 'swin_tiny':
            return ModelFactory.create_swin_tiny(num_classes, pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ----------------------------------------------------------------------
# 4. Trainer: handles training, validation, testing, logging, checkpoint
# ----------------------------------------------------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device,
                 criterion, optimizer, scheduler=None, exp_name="exp"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_name = exp_name

        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    def _validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Valid"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    def train(self, num_epochs, log_wandb=False, log_swanlab=False):
        for epoch in range(1, num_epochs+1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate()
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Logging
            if log_wandb and wandb_available and wandb.run is not None:
                wandb.log({
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            if log_swanlab and swanlab_available and swanlab.get_run() is not None:
                swanlab.log({
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, f"{self.exp_name}_best.pth")
                print(f"  -> Best model saved (val_acc={val_acc:.4f})")

        # Load best model for final test
        self.model.load_state_dict(self.best_model_wts)
        return self.history, self.best_acc

    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc


# ----------------------------------------------------------------------
# 5. Experiment Runner: manage multiple experiments with different configs
# ----------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, data_root, device, base_config, log_wandb=False, log_swanlab=False, csv_path="experiment_results.csv"):
        self.data_root = data_root
        self.device = device
        self.base_config = base_config
        self.log_wandb = log_wandb
        self.log_swanlab = log_swanlab
        self.csv_path = csv_path
        self.results = []
        if not os.path.exists(self.csv_path):
            self._write_csv_header()

    def _write_csv_header(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'exp_name', 'model_type', 'pretrained', 'lr', 'batch_size',
                'optimizer', 'weight_decay', 'use_scheduler', 'epochs',
                'best_val_acc', 'test_acc'
            ])

    def _append_result_to_csv(self, exp_name, config, best_val_acc, test_acc):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                exp_name,
                config.get('model_type', ''),
                config.get('pretrained', True),
                config.get('lr', self.base_config.get('lr', '')),
                config.get('batch_size', self.base_config.get('batch_size', '')),
                config.get('optimizer', self.base_config.get('optimizer', '')),
                config.get('weight_decay', self.base_config.get('weight_decay', '')),
                config.get('use_scheduler', self.base_config.get('use_scheduler', False)),
                config.get('epochs', self.base_config.get('epochs', '')),
                best_val_acc,
                test_acc
            ])

    def _exp_name(self, config):
        parts = [config['model_type']]
        if not config.get('pretrained', True):
            parts.append('scratch')
        parts.append(f"lr{config['lr']}")
        parts.append(f"bs{config.get('batch_size', self.base_config['batch_size'])}")
        if config.get('optimizer', 'adam') != 'adam':
            parts.append(config['optimizer'])
        return "_".join(str(p) for p in parts)

    def run_single(self, config):
        exp_name = self._exp_name(config)
        print(f"\n=== Starting experiment: {exp_name} ===")

        if self.log_wandb and wandb_available:
            wandb.init(project="flower102_task1", name=exp_name, config=config)
        if self.log_swanlab and swanlab_available:
            swanlab.init(project="flower102_task1", experiment_name=exp_name, config=config)

        batch_size = config.get('batch_size', self.base_config['batch_size'])
        data_module = FlowerDataModule(self.data_root, batch_size=batch_size)
        train_loader, val_loader, test_loader = data_module.get_loaders()
        num_classes = data_module.num_classes

        model = ModelFactory.create_model(
            model_type=config['model_type'],
            num_classes=num_classes,
            pretrained=config.get('pretrained', True)
        )

        criterion = nn.CrossEntropyLoss()
        lr = config.get('lr', self.base_config['lr'])
        weight_decay = config.get('weight_decay', 1e-4)
        opt_name = config.get('optimizer', 'adam').lower()
        if opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None
        if config.get('use_scheduler', False):
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        trainer = Trainer(
            model, train_loader, val_loader, test_loader, self.device,
            criterion, optimizer, scheduler, exp_name=exp_name
        )

        num_epochs = config.get('epochs', self.base_config['epochs'])
        history, best_val_acc = trainer.train(num_epochs, log_wandb=self.log_wandb, log_swanlab=self.log_swanlab)
        test_acc = trainer.test()

        self.results.append({
            'config': config,
            'exp_name': exp_name,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'history': history
        })

        # 保存到 CSV
        self._append_result_to_csv(exp_name, config, best_val_acc, test_acc)
        print(f"Results appended to {self.csv_path}")

        if self.log_wandb and wandb_available:
            wandb.finish()
        if self.log_swanlab and swanlab_available:
            swanlab.finish()

        return best_val_acc, test_acc

    def run_grid(self, config_list):
        for cfg in config_list:
            self.run_single(cfg)
        self.print_summary()
        print(f"\nAll results saved to {self.csv_path}")

    def print_summary(self):
        print("\n" + "="*60)
        print("Experiment Summary")
        print("="*60)
        for res in self.results:
            print(f"{res['exp_name']:<50} Val Acc: {res['best_val_acc']:.4f}  Test Acc: {res['test_acc']:.4f}")
        print("="*60)


# ----------------------------------------------------------------------
# 6. Main: parse args, define experiment grid, run
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Task1: Flower Classification")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root containing jpg/ (or images directly) and .mat files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--quick_test', action='store_true', help='Run only minimal configs for testing')
    parser.add_argument('--log_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_swanlab', action='store_true', help='Enable swanlab logging')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    set_seed(42)

    base_config = {
        'batch_size': 32,
        'epochs': 20,
        'lr': 1e-4,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'use_scheduler': False,
        'model_type': 'resnet18',
        'pretrained': True
    }

    runner = ExperimentRunner(
        data_root=args.data_root,
        device=device,
        base_config=base_config,
        log_wandb=args.log_wandb,
        log_swanlab=args.log_swanlab
    )

    if args.quick_test:
        # Quick test runs only a few epochs
        configs = [
            {'model_type': 'resnet18', 'pretrained': True, 'lr': 1e-4, 'epochs': 3, 'batch_size': 32},
            {'model_type': 'swin_tiny', 'pretrained': True, 'lr': 1e-4, 'epochs': 3, 'batch_size': 32}
        ]
        runner.run_grid(configs)
    else:
        # Full experiments as per assignment requirements
        configs = []

        # 1. Hyperparameter analysis: different learning rates for pretrained ResNet18
        for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            configs.append({
                'model_type': 'resnet18',
                'pretrained': True,
                'lr': lr,
                'epochs': 60,
                'batch_size': 32
            })

        # 2. Ablation: random initialization vs pretrained (best LR from above, say 1e-4 for pretrained, 1e-3 for scratch)
        configs.append({
            'model_type': 'resnet18',
            'pretrained': False,
            'lr': 1e-3,
            'epochs': 60,
            'batch_size': 32
        })
        # Also include pretrained baseline (already covered by lr=1e-4, but we ensure it's there)
        # No duplicate needed.

        # 3. Compare with Vision Transformer (Swin-Tiny) pretrained
        configs.append({
            'model_type': 'swin_tiny',
            'pretrained': True,
            'lr': 1e-4,
            'epochs': 60,
            'batch_size': 32
        })

        # Optionally, we can also add Swin-T from scratch, but not required.
        # configs.append({'model_type': 'swin_tiny', 'pretrained': False, 'lr': 1e-3, 'epochs': 20})

        runner.run_grid(configs)


if __name__ == '__main__':
    main()