"""
GAN Trainer for iSTFTNet vocoder: generator + discriminator, two optimizers.
"""

import torch
from torch.nn.utils import clip_grad_norm_

from src.trainer.base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    """
    Trainer for GAN-based vocoder (generator + discriminator).
    """

    def __init__(
        self,
        model,
        discriminator,
        criterion,
        metrics,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g=None,
        lr_scheduler_d=None,
        text_encoder=None,
        config=None,
        device=None,
        dataloaders=None,
        epoch_len=None,
        logger=None,
        writer=None,
        batch_transforms=None,
        skip_oom=True,
    ):
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer_g,
            lr_scheduler=lr_scheduler_g,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
            text_encoder=text_encoder,
        )

    def process_batch(self, batch, metrics):
        """
        One GAN step: forward generator, forward discriminator on real and pred,
        generator loss (mel + adv + fm), discriminator loss, backward both.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        mel_spec = batch["mel_spec"]
        waveform_target = batch.get("audio", batch.get("waveform"))
        if waveform_target.dim() == 2:
            waveform_target = waveform_target.unsqueeze(1)

        # generator forward
        model_kwargs = {k: v for k, v in batch.items() if k != "mel_spec"}
        waveform_pred = self.model(mel_spec, **model_kwargs)
        if waveform_pred.dim() == 2:
            waveform_pred = waveform_pred.unsqueeze(1)

        # discriminator on real and pred
        disc_real = self.discriminator(waveform_target)
        disc_pred = self.discriminator(waveform_pred.detach())

        real_outputs = disc_real["mpd_outputs"] + disc_real["msd_outputs"]
        pred_outputs = disc_pred["mpd_outputs"] + disc_pred["msd_outputs"]
        real_features = disc_real["mpd_features"] + disc_real["msd_features"]
        pred_features = disc_pred["mpd_features"] + disc_pred["msd_features"]

        # discriminator loss
        d_loss, _, _ = self.criterion.discriminator_loss(real_outputs, pred_outputs)
        if self.is_train:
            # backward, optimizer step only in train
            self.optimizer_d.zero_grad()
            d_loss.backward()
            if self.config["trainer"].get("max_grad_norm_d") is not None:
                clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.config["trainer"]["max_grad_norm_d"],
                )
            self.optimizer_d.step()

        # generator loss: run discriminator again on pred (no detach)
        disc_pred_g = self.discriminator(waveform_pred)
        pred_outputs_g = disc_pred_g["mpd_outputs"] + disc_pred_g["msd_outputs"]
        pred_features_g = disc_pred_g["mpd_features"] + disc_pred_g["msd_features"]

        waveform_pred_flat = waveform_pred.squeeze(1)
        waveform_target_flat = waveform_target.squeeze(1)

        # model output is longer than target
        target_len = waveform_target_flat.shape[-1]
        waveform_pred_flat = waveform_pred_flat[..., :target_len]

        gen_losses = self.criterion(
            waveform_pred_flat,
            waveform_target_flat,
            discriminator_outputs=pred_outputs_g,
            discriminator_features_pred=pred_features_g,
            discriminator_features_target=[f.detach() for f in real_features],
            **batch,
        )
        g_loss = gen_losses["loss"]
        if self.is_train:
            self.optimizer_g.zero_grad()
            g_loss.backward()
            self._clip_grad_norm()
            self.optimizer_g.step()

        # build return batch for logging
        out_batch = {
            **batch,
            "loss": gen_losses["loss"],
            "mel_loss": gen_losses.get("mel_loss", torch.tensor(0.0, device=self.device)),
            "adv_loss": gen_losses.get("adv_loss", torch.tensor(0.0, device=self.device)),
            "fm_loss": gen_losses.get("fm_loss", torch.tensor(0.0, device=self.device)),
            "d_loss": d_loss,
            "g_loss": g_loss,
            "waveform_pred": waveform_pred_flat,
        }
        if "audio" not in out_batch and "waveform" in batch:
            out_batch["audio"] = batch["waveform"]

        for name in ["loss", "mel_loss", "adv_loss", "fm_loss", "d_loss"]:
            if name in out_batch and out_batch[name].dim() == 0:
                metrics.update(name, out_batch[name].item())

        return out_batch

    def _log_batch(self, batch_idx, batch, mode="train", epoch=None):
        """Log scalars and audio (target + predicted) when available."""
        if self.writer is None:
            return
        if "waveform_pred" not in batch:
            return
        sample_rate = self.config.get("loss_function", {}).get("sample_rate", 22050)
        pred = batch["waveform_pred"]
        if pred.dim() > 1:
            pred = pred[0]
        self.writer.add_audio("pred", pred, sample_rate=sample_rate)
        target = batch.get("audio", batch.get("waveform"))
        if target is not None:
            t = target[0] if target.dim() > 1 else target
            self.writer.add_audio("target", t, sample_rate=sample_rate)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """Save checkpoint with both generator and discriminator optimizers/schedulers."""
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if self.lr_scheduler_g is not None:
            state["lr_scheduler"] = self.lr_scheduler_g.state_dict()
        if self.lr_scheduler_d is not None:
            state["lr_scheduler_d"] = self.lr_scheduler_d.state_dict()

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """Resume from checkpoint (generator, discriminator, both optimizers/schedulers)."""
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration differs from checkpoint."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        cfg_opt = checkpoint["config"].get("optimizer_g"), checkpoint["config"].get("optimizer_d")
        cur_opt = self.config.get("optimizer_g"), self.config.get("optimizer_d")
        if cfg_opt != cur_opt:
            self.logger.warning(
                "Warning: Optimizer config differs; optimizer state not resumed."
            )
        else:
            self.optimizer_g.load_state_dict(checkpoint["optimizer"])
            if "optimizer_d" in checkpoint:
                self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            if self.lr_scheduler_g is not None and "lr_scheduler" in checkpoint:
                self.lr_scheduler_g.load_state_dict(checkpoint["lr_scheduler"])
            if self.lr_scheduler_d is not None and "lr_scheduler_d" in checkpoint:
                self.lr_scheduler_d.load_state_dict(checkpoint["lr_scheduler_d"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )
