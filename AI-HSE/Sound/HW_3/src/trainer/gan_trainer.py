import torch
from torch.nn.utils import clip_grad_norm_
from src.logger.utils import plot_spectrogram

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    """
    Trainer for GAN-based vocoder
    Two-step training: discriminator step, then generator step.
    """

    def __init__(
        self,
        model,  # gen
        discriminator,
        criterion,  # iSTFTLoss
        metrics,
        optimizer_g,  # gen optimizer
        optimizer_d,  # dis optimizer
        lr_scheduler_g=None,
        lr_scheduler_d=None,
        text_encoder=None,
        config=None,
        device="cuda",
        dataloaders=None,
        logger=None,
        writer=None,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model: Generator model
            discriminator: Discriminator model
            criterion: Loss function
            optimizer_g: Optimizer for generator
            optimizer_d: Optimizer for discriminator
            lr_scheduler_g: LR scheduler for generator (optional)
            lr_scheduler_d: LR scheduler for discriminator (optional)
            ... (other args same as BaseTrainer)
        """

        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer_g,
            lr_scheduler=lr_scheduler_g,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )
        
        self.discriminator = discriminator.to(device)
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d

    def process_batch(self, batch, metrics: MetricTracker):
        """
        GAN training: D step first, then G step.
        
        Args:
            batch (dict): contain:
                - mel_spec or spectrogram: [B, C, T] mel-spectrogram
                - audio or waveform: [B, L] target waveform
            metrics (MetricTracker): Metric tracker
        Returns:
            batch (dict): Updated with outputs and losses
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)


        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # get inputs (mel-spectrogram, target waveform)
        mel_spec = batch.get("mel_spec")
        if mel_spec is None:
            mel_spec = batch.get("spectrogram")
        waveform_target = batch.get("waveform")
        if waveform_target is None:
            waveform_target = batch.get("audio")
        
        # Ensure waveform_target is 2D [B, L]
        if waveform_target.dim() == 3:
            waveform_target = waveform_target.squeeze(1)  # [B, 1, L] -> [B, L]
        
        # 1. Generator forward
        waveform_pred = self.model(mel_spec)  # [B, L]
        
        # 2. Discriminator step: D(real) and D(fake.detach())
        self.discriminator.train()
        out_real = self.discriminator(waveform_target)
        out_fake_detached = self.discriminator(waveform_pred.detach())
        
        disc_real_outputs = out_real["mpd_outputs"] + out_real["msd_outputs"]
        disc_fake_outputs = out_fake_detached["mpd_outputs"] + out_fake_detached["msd_outputs"]
        
        d_loss, r_losses, g_losses = self.criterion.discriminator_loss(
            disc_real_outputs, disc_fake_outputs
        )
        
        # Save g_loss (sum of g_losses) for logging
        g_loss_sum = sum(g_losses) if g_losses else torch.tensor(0.0, device=d_loss.device)
        
        if self.is_train:
            # D backward
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self._clip_grad_norm_d()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()
        
        # 3. Generator step: D(waveform_pred) with grad, then G loss
        out_fake = self.discriminator(waveform_pred)
        disc_outputs_fake = out_fake["mpd_outputs"] + out_fake["msd_outputs"]
        feat_fake = out_fake["mpd_features"] + out_fake["msd_features"]
        feat_real = [f.detach() for f in out_real["mpd_features"] + out_real["msd_features"]]
        
        losses = self.criterion(
            waveform_pred=waveform_pred,
            waveform_target=waveform_target,
            discriminator_outputs=disc_outputs_fake,
            discriminator_features_pred=feat_fake,
            discriminator_features_target=feat_real,
        )
        
        if self.is_train:
            # G backward
            self.optimizer_g.zero_grad()
            losses["loss"].backward()
            self._clip_grad_norm_g()
            self.optimizer_g.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()
        
        # update batch with outputs and losses
        batch.update({
            "waveform_pred": waveform_pred,
            "waveform_target": waveform_target,
            "mel_spec": mel_spec,
        })
        batch.update(losses)
        batch["d_loss"] = d_loss
        batch["g_loss"] = g_loss_sum if isinstance(g_loss_sum, torch.Tensor) else torch.tensor(g_loss_sum, device=d_loss.device)
        
        # update metrics
        loss_names = self.config.writer.get("loss_names", ["loss", "mel_loss", "adv_loss", "fm_loss", "d_loss"])
        for loss_name in loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())
        
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        
        return batch

    def _clip_grad_norm_g(self):
        """Clip gradients for generator."""
        if self.cfg_trainer.get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(),
                self.cfg_trainer.max_grad_norm
            )

    def _clip_grad_norm_d(self):
        """Clip gradients for discriminator."""
        if self.cfg_trainer.get("max_grad_norm_d", None) is not None:
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg_trainer.max_grad_norm_d
            )
        elif self.cfg_trainer.get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg_trainer.max_grad_norm
            )

    def _log_batch(self, batch_idx, batch, mode="train"):
        """Log audio/spectrograms for vocoder."""
        if mode == "train" and batch_idx % self.log_step == 0:
            self.log_spectrogram(**batch)
        else:
            self.log_spectrogram(**batch)

    def log_spectrogram(self, mel_spec=None, spectrogram=None, **batch):
        """Log mel-spectrogram."""
        spec = mel_spec or spectrogram
        if spec is not None:
            spec_for_plot = spec[0].detach().cpu()
            image = plot_spectrogram(spec_for_plot)
            self.writer.add_image("mel_spectrogram", image)
