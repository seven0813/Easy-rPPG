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

    SUPPORTED_TEST_PROTOCOLS = ("intra", "cross")

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
        self.test_protocols = self._resolve_test_protocols()

        self.setup_mode()
        self.resume_state()

    def _resolve_test_protocols(self):
        """Validate the optional train.test_protocols protocol list."""
        if self.config.mode != "train_and_test":
            return None

        configured = self.config.train.get("test_protocols")
        if configured is None:
            return None
        if not isinstance(configured, (list, tuple)):
            raise TypeError("train.test_protocols must be a non-empty list")
        if not configured:
            raise ValueError("train.test_protocols must not be empty")

        protocols = []
        for value in configured:
            if not isinstance(value, str):
                raise TypeError("Each train.test_protocols item must be a string")
            protocol = value.strip().lower()
            if protocol not in self.SUPPORTED_TEST_PROTOCOLS:
                supported = ", ".join(self.SUPPORTED_TEST_PROTOCOLS)
                raise ValueError(
                    f"Unsupported test protocol {value!r}. Supported: {supported}"
                )
            if protocol in protocols:
                raise ValueError(
                    f"train.test_protocols contains duplicate protocol {protocol!r}"
                )
            protocols.append(protocol)
        return tuple(protocols)

    @property
    def uses_protocol_mode(self):
        return self.test_protocols is not None

    @property
    def uses_validation(self):
        if self.uses_protocol_mode:
            return "cross" in self.test_protocols
        return not self.config.train.use_last_epoch

    def _validate_protocol_datasets(self):
        if not self.uses_protocol_mode:
            return
        required = {"train", "test_intra"}
        if "cross" in self.test_protocols:
            required.add("test_cross")
        missing = sorted(required.difference(self.data_loader))
        if missing:
            raise ValueError(
                "train.test_protocols requires dataset loaders: "
                f"{', '.join(missing)}"
            )

    def _validation_loader(self, data_loader):
        if not self.uses_validation:
            return None
        dataset_key = "test_intra" if self.uses_protocol_mode else "val"
        if dataset_key not in data_loader:
            raise ValueError(
                f"Validation requires dataset loader {dataset_key!r}"
            )
        return data_loader[dataset_key]

    def _update_best_model(self, valid_loss, epoch):
        if (
            self.min_valid_loss is not None
            and valid_loss >= self.min_valid_loss
        ):
            return False
        self.min_valid_loss = valid_loss
        self.best_epoch = epoch
        self.logger.info(
            "Update best model! Best epoch: %s", self.best_epoch
        )
        self.save_best_model(epoch)
        return True


    def setup_mode(self):
        if self.config.mode == "train_and_test":
            self._validate_protocol_datasets()
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
        validation_loader = self._validation_loader(data_loader)
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
                # print("imgs shape:", imgs.shape)

                ## 1. forward
                pred_bvp, _,_,_ = self.model(imgs)              ### pred_bvp: [B, T]
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

            if validation_loader is not None:
                valid_loss = self.valid(validation_loader)
                mean_valid_losses.append(valid_loss)

                self.writer.add_scalar("valid/loss", valid_loss, epoch)
                self.logger.info("validation loss: %s", valid_loss)

                self._update_best_model(valid_loss, epoch)

            self.save_model(epoch)

        if self.uses_validation:
            self.logger.info(
                "best trained epoch: %s, min_val_loss: %s",
                self.best_epoch,
                self.min_valid_loss,
            )

        if self.config.train.plot_losses_and_lr:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

        self.writer.close()
        self.writer = None

    def valid(self, valid_loader):
        self.logger.info("====Validating====")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(valid_loader, dynamic_ncols=True)
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


    def _checkpoint_path(self, checkpoint_name):
        if checkpoint_name not in {"latest", "best"}:
            raise ValueError(
                "checkpoint_name must be either 'latest' or 'best'"
            )
        return os.path.join(self.model_ckpt_dir, f"{checkpoint_name}.pth")

    def _load_test_checkpoint(self, checkpoint_path, checkpoint_label):
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"PhysNet {checkpoint_label} checkpoint does not exist: "
                f"{checkpoint_path}"
            )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.logger.info(
            "Testing uses %s checkpoint: %s",
            checkpoint_label,
            checkpoint_path,
        )

    def _test_once(
        self,
        test_loader,
        dataset_key,
        checkpoint_path,
        checkpoint_label,
        protocol_label,
        output_filename="outputs.pickle",
    ):
        self.logger.info("")
        self.logger.info(
            "===Testing protocol=%s dataset=%s checkpoint=%s===",
            protocol_label,
            dataset_key,
            checkpoint_label,
        )
        self._load_test_checkpoint(checkpoint_path, checkpoint_label)

        predictions = {}
        labels = {}
        self.model.eval()
        with torch.no_grad():
            for test_batch in tqdm(test_loader, dynamic_ncols=True):
                test_imgs = test_batch[0].to(
                    self.device, dtype=torch.float32
                )
                test_bvp = test_batch[1].to(self.device)
                pred_bvp, _, _, _ = self.model(test_imgs)

                test_bvp = test_bvp.cpu()
                pred_bvp = pred_bvp.cpu()
                for index in range(test_imgs.shape[0]):
                    subject = test_batch[2][index]
                    clip_index = int(test_batch[3][index])
                    predictions.setdefault(subject, {})[clip_index] = pred_bvp[
                        index
                    ]
                    labels.setdefault(subject, {})[clip_index] = test_bvp[index]

        self.logger.info(
            "Evaluation protocol=%s, method=%s",
            protocol_label,
            self.config.inference.eval_method,
        )
        metrics = calculate_metrics(predictions, labels, self.config)
        self.log_test_metrics(metrics)
        self.save_test_outputs(
            predictions,
            labels,
            self.config,
            metrics,
            output_filename=output_filename,
        )
        return metrics

    def test(self, data_loader):
        if self.config.mode == "only_test":
            return self._test_once(
                data_loader["test"],
                "test",
                self.pretrain_ckpt,
                "pretrained",
                "only_test",
            )

        if not self.uses_protocol_mode:
            checkpoint_name = (
                "latest" if self.config.train.use_last_epoch else "best"
            )
            return self._test_once(
                data_loader["test"],
                "test",
                self._checkpoint_path(checkpoint_name),
                checkpoint_name,
                "legacy",
            )

        results = {}
        combined = len(self.test_protocols) == 2
        if "intra" in self.test_protocols:
            results["intra"] = self._test_once(
                data_loader["test_intra"],
                "test_intra",
                self._checkpoint_path("latest"),
                "latest",
                "intra",
                output_filename=(
                    "outputs_intra.pickle" if combined else "outputs.pickle"
                ),
            )
        if "cross" in self.test_protocols:
            results["cross"] = self._test_once(
                data_loader["test_cross"],
                "test_cross",
                self._checkpoint_path("best"),
                "best",
                "cross",
            )
        return results


