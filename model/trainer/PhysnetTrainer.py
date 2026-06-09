"""PhysNet Trainer."""
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from evaluate.metrics import calculate_metrics
from model.loss.PhysNetNegPearsonLoss import Neg_Pearson
from model.network.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from model.trainer.BaseTrainer import BaseTrainer
from model.utils import get_root_logger
from tqdm import tqdm



class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.device)
        self.vis_dir = config.path.visualization
        self.num_of_gpu = config.num_gpu
        self.fs = config.datasets.fs
        self.data_loader = data_loader
        self.config = config
        
        self.logger = get_root_logger()

        self.setup_mode()
        self.resume_state()
    

    
    def setup_mode(self):
        if self.config.mode == "train_and_test":
            self.train_clip_length = self.config.datasets.train.clip_length
            self.max_epoch_num = self.config.train.epochs 
            self.model_ckpt_dir = self.config.path.models
            self.training_state_dir = self.config.path.training_states

            self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=self.train_clip_length).to(self.device)  # [3, T, 128,128]
            self.min_valid_loss = None
            self.best_epoch = 0
            self.writer = None
            self.num_train_batches = len(self.data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.train.lr)

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.train.lr, 
                epochs=self.config.train.epochs, 
                steps_per_epoch=self.num_train_batches)
            self.writer = SummaryWriter(log_dir=self.config.path.tensorboard)

        elif self.config.mode == "only_test":
            clip_length = self.config.datasets.test.clip_length
            self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=clip_length).to(self.device)  # [3, T, 128,128]
            self.pretrain_ckpt = self.config.inference.pretrain_ckpt
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")


    def train(self, data_loader):
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.start_epoch, self.max_epoch_num):
            self.logger.info('')
            self.logger.info("====Training Epoch: %s====", epoch)
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], dynamic_ncols=True)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                ## 0. prepare data
                ### imgs:BCTHW [0-1]   bvp: BT [float32]
                imgs = batch[0].to(torch.float32).to(self.device)
                bvp = batch[1].to(torch.float32).to(self.device)
                
                print('imgs.shape', imgs.shape)
                print('bvp.shape', bvp.shape)
                
                ## 1. forward
                pred_bvp, _,_,_ = self.model(imgs)              ### pred_bvp: [B, T]
                print('pred_bvp.shape', pred_bvp.shape)
                pred_bvp = (pred_bvp - torch.mean(pred_bvp)) / torch.std(pred_bvp)  # normalize
                bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize    
                
                ## 2. backward
                loss = self.loss_model(pred_bvp, bvp)
                loss.backward()
                
                # logging
                train_loss.append(loss.item())
                current_lr = self.scheduler.get_last_lr()[0]
                lrs.append(current_lr)

                global_step = epoch * self.num_train_batches + idx
                self.writer.add_scalar("train/loss", loss.item(), global_step)
                self.writer.add_scalar("train/lr", current_lr, global_step)

                ## optimize
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item(), lr=current_lr)

            # Append the mean training loss for the epoch
            mean_train_loss = np.mean(train_loss)
            mean_training_losses.append(mean_train_loss)
            self.writer.add_scalar("train/loss_epoch", mean_train_loss, epoch)

            ## 如果 use_last_epoch = False，则进行验证选择 best model
            if not self.config.train.use_last_epoch: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                
                self.writer.add_scalar("valid/loss", valid_loss, epoch)
                self.logger.info("validation loss: %s", valid_loss)
                
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.logger.info("Update best model! Best epoch: %s", self.best_epoch)
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.logger.info("Update best model! Best epoch: %s", self.best_epoch)
            
            self.save_model(epoch)
        
        if not self.config.train.use_last_epoch: 
            self.logger.info(
                "best trained epoch: %s, min_val_loss: %s",
                self.best_epoch,
                self.min_valid_loss,
            )
            
        if self.config.train.plot_losses_and_lr:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
    
        self.writer.close()
        self.writer = None

    def valid(self, data_loader):
        self.logger.info("====Validating====")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["val"], dynamic_ncols=True)
            for _, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                
                ## 0. prepare data
                val_imgs = valid_batch[0].to(torch.float32).to(self.device)
                val_bvp = valid_batch[1].to(torch.float32).to(self.device)
                
                ## 1. forward
                pred_bvp, _,_,_ = self.model(val_imgs)
                val_bvp = (val_bvp - torch.mean(val_bvp)) / torch.std(val_bvp)  # normalize
                pred_bvp = (pred_bvp - torch.mean(pred_bvp)) / torch.std(pred_bvp)  # normalize

                ## 2. calculate loss
                loss_ecg = self.loss_model(val_bvp, pred_bvp)
                valid_loss.append(loss_ecg.item())
                
                ## logging
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)


    def test(self, data_loader):    
        self.logger.info('')    
        self.logger.info("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.mode == "only_test":
            if not os.path.exists(self.pretrain_ckpt):
                raise ValueError("Inference model path error! Please check inference.pretrain_ckpt in your yaml.")
            self.model.load_state_dict(torch.load(self.pretrain_ckpt))
            self.logger.info("Testing uses pretrained model: %s", self.pretrain_ckpt)
        else:
            ##! test_dataloader的clip_length可以和train/val不一样
            test_clip_length = self.config.datasets.test.clip_length
            if test_clip_length != self.train_clip_length:
                self.model.poolspa = torch.nn.AdaptiveAvgPool3d((test_clip_length, 1, 1)).to(self.device)
                self.logger.info("Test clip length changed to: %s", test_clip_length)
            if self.config.train.use_last_epoch:
                last_epoch_model_path = os.path.join(self.model_ckpt_dir, f"Epoch{self.max_epoch_num - 1}.pth")
                self.logger.info("Testing uses last epoch ckpt: %s",last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(self.model_ckpt_dir, f"Epoch{self.best_epoch}.pth")
                self.logger.info("Testing uses best epoch selected using model selection: %s",best_model_path,)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], dynamic_ncols=True)):
                
                test_imgs = test_batch[0].to(self.device)
                batch_size = test_imgs.shape[0]
                test_bvp = test_batch[1].to(self.device)
                
                pred_bvp,_,_,_ = self.model(test_imgs)

                test_bvp = test_bvp.cpu()
                pred_bvp = pred_bvp.cpu()

                ## 把同一个 subject 的不同 clip 的预测结果和标签放在一起，方便后续计算指标
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_bvp[idx]
                    labels[subj_index][sort_index] = test_bvp[idx]
                
        ME, STD, MAE, RMSE, MER, P = calculate_metrics(predictions, labels, self.config)
        self.logger.info(
            "\n"
            "ME:   %.4f\n"
            "STD:  %.4f\n"
            "MAE:  %.4f\n"
            "RMSE: %.4f\n"
            "MER:  %.4f\n"
            "P:    %.4f",
            ME, STD, MAE, RMSE, MER, P,
            )        
        self.save_test_outputs(predictions, labels, self.config)



    
