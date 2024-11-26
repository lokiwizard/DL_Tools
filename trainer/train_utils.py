from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, config_path):

        """
        :param model: 模型
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param config_path: 配置文件路径
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = OmegaConf.load(config_path)
        self._setup()

    def _setup(self):
        # 设置训练参数
        self.model_name = self.config.model_name  # 模型名称
        self.learning_rate = self.config.training.learning_rate  # 学习率
        self.num_epochs = self.config.training.num_epochs  # 训练的总轮数
        self.use_amp = self.config.training.get('use_amp', False)  # 是否使用混合精度训练
        self.dynamic_loss_adjustment = self.config.training.get('dynamic_loss_adjustment', False)  # 是否动态调整损失函数
        self.save_dir = self.config.training.get('save_dir', './saved_models')  # 保存模型的目录
        self.pretrained = self.config.training.get('pretrained', False)  # 是否加载预训练模型
        self.pretrained_model_path = self.config.training.get('pretrained_model_path', None)  # 预训练模型路径

        # 创建保存模型的目录
        os.makedirs(self.save_dir, exist_ok=True)  # 如果目录不存在则创建

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 如果需要加载预训练模型
        if self.pretrained:
            if self.pretrained_model_path is None:
                raise ValueError("Pretrained model path must be specified when 'pretrained' is True.")
            else:
                self.load_pretrained_model(self.pretrained_model_path)

        # 初始化优化器
        self._init_optimizer()

        # 初始化损失函数
        self._init_loss_function()

        # 初始化混合精度训练的 GradScaler
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()

        # 初始化学习率调度器
        self._init_scheduler()

        # 初始化用于保存最佳模型的变量
        self.best_val_accuracy = 0.0
        self.best_model_wts = None

    def _init_optimizer(self):

        """
        Adam, SGD, AdamW 三种优化器
        Adam的可选参数有：betas, eps, weight_decay
        SGD的可选参数有：momentum, weight_decay
        AdamW的可选参数有：betas, eps, weight_decay
        """

        optimizer_config = self.config.optimizer
        optimizer_type = optimizer_config.type
        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate
        }
        # 添加可选的优化器参数
        if 'weight_decay' in optimizer_config:
            optimizer_params['weight_decay'] = optimizer_config.weight_decay
        if 'momentum' in optimizer_config:
            optimizer_params['momentum'] = optimizer_config.momentum
        if 'betas' in optimizer_config:
            optimizer_params['betas'] = optimizer_config.betas
        if 'eps' in optimizer_config:
            optimizer_params['eps'] = optimizer_config.eps

        # 根据优化器类型初始化
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(**optimizer_params)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(**optimizer_params)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(**optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _init_loss_function(self):
        loss_config = self.config.loss
        loss_type = loss_config.type
        if loss_type == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss_type == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif loss_type == 'BCELoss':
            self.criterion = torch.nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def _init_scheduler(self):
        self.scheduler = None
        if 'scheduler' in self.config and self.config.scheduler.type is not None:
            scheduler_type = self.config.scheduler.type
            scheduler_config = self.config.scheduler
            if scheduler_type == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.step_size,
                    gamma=scheduler_config.gamma
                )
            elif scheduler_type == 'MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=scheduler_config.milestones,
                    gamma=scheduler_config.gamma
                )
            elif scheduler_type == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=scheduler_config.gamma
                )
            elif scheduler_type == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.T_max,
                    eta_min=scheduler_config.get('eta_min', 0)
                )
            elif scheduler_type == 'CosineAnnealingWarmRestarts':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=scheduler_config.T_0,
                    T_mult=scheduler_config.get('T_mult', 1),
                    eta_min=scheduler_config.get('eta_min', 0)
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def load_pretrained_model(self, model_path):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pretrained model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Pretrained model file not found at {model_path}")

    def train(self):
        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training")
            for inputs, targets in train_loader_tqdm:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(enabled=self.use_amp, device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                train_loader_tqdm.set_postfix(loss=loss.item())

                # 动态调整损失函数
                if self.dynamic_loss_adjustment:
                    self.adjust_loss_function(epoch, loss)

            avg_train_loss = train_loss / total
            train_accuracy = correct / total

            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch + 1)
                else:
                    self.scheduler.step()

            # 验证阶段
            val_loss, val_accuracy = self.validate()

            # 保存最佳模型权重
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_wts = self.model.state_dict()
                self.save_best_model()

            print(f"Epoch {epoch + 1}/{self.num_epochs} completed.")
            print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.4f}%")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.4f}%\n")



    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(self.val_loader, desc="Validation")
            for inputs, targets in val_loader_tqdm:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.amp.autocast(enabled=self.use_amp, device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / total
        val_accuracy = correct / total

        return avg_val_loss, val_accuracy

    def adjust_loss_function(self, epoch, loss):
        # 实现您的损失函数动态调整逻辑
        pass

    def save_best_model(self):
        save_path = os.path.join(self.save_dir, f"{self.model_name}_best_checkpoint.pth")
        torch.save(self.best_model_wts, save_path)
        print(f"Best model saved to {save_path}")

"""
usage:
def main():
    train_loader, val_loader, num_classes = get_animal_data_loader(root_dir=r'dataset_dir',
                                                                  batch_size=128,
                                                                 image_size=224)

    model = resnet18(num_classes=num_classes)

    trainer = Trainer(model, train_loader, val_loader, config_path=r"config.yaml")

    trainer.train()

"""
