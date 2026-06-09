import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import pickle


class BaseTrainer:
    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass
    
    def resume_state(self):
        """Resume model and training state from config.path.resume_state."""
        self.start_epoch = 0
        if self.config.mode != "train_and_test":
            return

        resume_path = self.config.train.get("resume_state")
        if not resume_path:
            return
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume state does not exist: {resume_path}")

        state = torch.load(resume_path, map_location="cpu")
        required_keys = {"epoch", "model_path", "optimizer", "scheduler"}
        missing_keys = required_keys.difference(state)
        if missing_keys:
            raise KeyError(
                f"Resume state {resume_path} is missing keys: {sorted(missing_keys)}"
            )

        model_path = state["model_path"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Resume model checkpoint does not exist: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self._restore_optimizer_state_devices()
        self.scheduler.load_state_dict(state["scheduler"])
        self.start_epoch = int(state["epoch"]) + 1
        self.best_epoch = int(state.get("best_epoch", 0))
        self.min_valid_loss = state.get("min_valid_loss")
        if self.start_epoch >= self.max_epoch_num:
            raise ValueError(
                f"Resume state has completed epoch {state['epoch']}, but "
                f"train.epochs is {self.max_epoch_num}. Set train.epochs greater "
                f"than {self.start_epoch} to continue training."
            )

        self.logger.info(
            "Resumed training from state: %s, next epoch: %s",
            resume_path,
            self.start_epoch,
        )

    def _restore_optimizer_state_devices(self):
        """Restore optimizer tensors to devices expected by the current Adam."""
        capturable = bool(self.optimizer.defaults.get("capturable", False))
        for param_state in self.optimizer.state.values():
            for key, value in param_state.items():
                if not torch.is_tensor(value):
                    continue
                if key == "step" and not capturable:
                    param_state[key] = value.cpu()
                else:
                    param_state[key] = value.to(self.device)
    
    
    def save_model(self, index):
        os.makedirs(self.model_ckpt_dir, exist_ok=True)
        os.makedirs(self.training_state_dir, exist_ok=True)

        model_path = os.path.join(self.model_ckpt_dir, f"Epoch{index}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info("Saved model path: %s", model_path)

        state_path = os.path.join(self.training_state_dir, f"Epoch{index}.state")
        state = {
            "epoch": index,
            "model_path": model_path,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_epoch": self.best_epoch,
            "min_valid_loss": self.min_valid_loss
        }
        torch.save(state, state_path)
        self.logger.info("Saved training state path: %s", state_path)
    
    
    def save_test_outputs(self, predictions, labels, config):
        
        if config.mode == 'train_and_test':
            output_dir = config.path.experiments_root
        elif config.mode == 'only_test':
            output_dir = config.path.results_root
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, 'outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.datasets.label_type
        data['fs'] = config.datasets.fs

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f'Saving outputs to:{output_path}')

    def plot_losses_and_lrs(self, train_losses, valid_losses, lrs, config):
        """
        train_losses: List[float]
        valid_losses: List[float] (可为空)
        lrs:          List[float]  (标量)
        保存 loss 曲线、learning rate 曲线及 CSV
        """
        import os
        import csv
        import matplotlib
        matplotlib.use("Agg")  # 服务器/无显示环境
        import matplotlib.pyplot as plt
    
        save_dir = self.vis_dir
        os.makedirs(save_dir, exist_ok=True)
    
        # 写 CSV
        csv_path = os.path.join(save_dir, "loss_lr_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "valid_loss", "lr"])
            # 步数按训练 loss 长度来
            steps = max(len(train_losses), len(lrs))
            for i in range(steps):
                tr = train_losses[i] if i < len(train_losses) else ""
                va = valid_losses[i] if i < len(valid_losses) else ""
                lr = lrs[i] if i < len(lrs) else ""
                writer.writerow([i, tr, va, lr])
    
        # 画 loss 图
        plt.figure(figsize=(8, 5))
        if len(train_losses) > 0:
            plt.plot(train_losses, label="train loss", linewidth=2)
        if len(valid_losses) > 0:
            plt.plot(valid_losses, label="valid loss", linewidth=2)
        if len(train_losses) > 0 or len(valid_losses) > 0:
            plt.legend(loc="best")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Loss curves")
        loss_png = os.path.join(save_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_png, dpi=200)
        plt.close()

        # 画 learning rate 图
        lr_png = os.path.join(save_dir, "lr_curve.png")
        if len(lrs) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(lrs, label="lr", linewidth=2)
            plt.xlabel("step")
            plt.ylabel("learning rate")
            plt.title("Learning rate curve")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(lr_png, dpi=200)
            plt.close()

        self.logger.info("Plot loss and lr curves to {}".format(save_dir))
