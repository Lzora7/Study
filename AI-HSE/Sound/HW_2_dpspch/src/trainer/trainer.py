from pathlib import Path

import pandas as pd
import torch

from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)
        
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        
        # Check for NaN loss before backward pass
        if torch.isnan(batch["loss"]) or torch.isinf(batch["loss"]):
            self.logger.warning(f"NaN/Inf loss detected: {batch['loss'].item()}. Skipping batch.")
            # Zero gradients to prevent issues
            self.optimizer.zero_grad()
            # Return zero loss for metrics
            batch["loss"] = torch.tensor(0.0, device=batch["loss"].device, requires_grad=False)
            return batch

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            
            # Check for NaN gradients before clipping
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self.logger.warning(f"NaN/Inf gradient detected in {name}. Skipping optimizer step.")
                        has_nan_grad = True
                        break
            
            if not has_nan_grad:
                # Log detailed gradient statistics before clipping
                grad_stats = self._get_gradient_statistics()
                
                # Log gradient statistics periodically (use batch_idx from base_trainer)
                # We'll log in _train_epoch where batch_idx is available
                
                self._clip_grad_norm()
                self.optimizer.step()
            else:
                # Zero gradients if NaN detected
                self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            # Only log predictions if required keys are present
            if "text" in batch and "log_probs" in batch and "log_probs_length" in batch:
                self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        # Логирование спектрограмм отключено
        pass

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path=None, audio=None, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        
        # Handle optional audio_path and audio
        if audio_path is None:
            audio_path = batch.get("audio_path", [""] * len(text))
        if audio is None:
            audio = batch.get("audio", [None] * len(text))
        
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path, audio))

        rows = {}
        for i, (pred, target, raw_pred, audio_path, audio_tensor) in enumerate(tuples[:examples_to_log]):
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            audio_name = Path(audio_path).name if audio_path else f"example_{i}"
            rows[audio_name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
            
            # Log audio to CometML if available
            if audio_tensor is not None:
                # audio_tensor shape: [1, T] or [T], sample_rate is 16000
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.squeeze(0)  # [T]
                self.writer.add_audio(
                    audio_name=f"audio_{i}_{audio_name}",
                    audio=audio_tensor,
                    sample_rate=16000,
                )
            
            # Log text predictions
            self.writer.add_text(
                text_name=f"prediction_{i}_{audio_name}",
                text=f"Target: {target}\nPrediction: {pred}\nWER: {wer:.2f}%\nCER: {cer:.2f}%",
            )
            
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
    
    def log_prediction_examples(
        self, text, log_probs, log_probs_length, num_examples=3, **batch
    ):
        """
        Log prediction examples to text logger.
        
        Args:
            text (list): list of target text strings
            log_probs (Tensor): log probabilities from model
            log_probs_length (Tensor): actual sequence lengths
            num_examples (int): number of examples to log
        """
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        
        self.logger.info("Prediction examples:")
        for i, (pred, target) in enumerate(zip(argmax_texts[:num_examples], text[:num_examples])):
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            self.logger.info(f"  Example {i+1}:")
            self.logger.info(f"    Target:  {target}")
            self.logger.info(f"    Predict: {pred}")
            self.logger.info(f"    WER: {wer:.2f}%, CER: {cer:.2f}%")
    
    @torch.no_grad()
    def _get_gradient_statistics(self):
        """
        Collect detailed gradient statistics for all model parameters.
        
        Returns:
            dict: Dictionary with gradient statistics for each parameter
        """
        grad_stats = {}
        zero_grad_count = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'abs_mean': grad.abs().mean().item(),
                    'norm': grad.norm().item(),
                    'numel': grad.numel(),
                }
            else:
                zero_grad_count += 1
                grad_stats[name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'abs_mean': 0.0,
                    'norm': 0.0,
                    'numel': 0,
                    'zero_grad': True
                }
        
        grad_stats['_summary'] = {
            'zero_grad_count': zero_grad_count,
            'total_params': total_params,
            'zero_grad_ratio': zero_grad_count / total_params if total_params > 0 else 0.0
        }
        
        return grad_stats
    
    def _log_gradient_statistics(self, grad_stats):
        """
        Log gradient statistics to logger and writer.
        
        Args:
            grad_stats (dict): Dictionary with gradient statistics
        """
        summary = grad_stats.get('_summary', {})
        zero_grad_ratio = summary.get('zero_grad_ratio', 0.0)
        
        # Log summary
        self.logger.info(
            f"Gradient statistics: {summary.get('zero_grad_count', 0)}/{summary.get('total_params', 0)} "
            f"parameters have zero gradients ({zero_grad_ratio:.1%})"
        )
        
        # Key layers to monitor
        key_layers = [
            'conv_layers',
            'rnn',
            'output_projection',
        ]
        
        # Collect statistics for key layers
        key_stats = {}
        for key in key_layers:
            for name, stats in grad_stats.items():
                if name == '_summary':
                    continue
                if key in name and not stats.get('zero_grad', False):
                    key_stats[name] = stats
                    break
        
        # Log key layer statistics
        if key_stats:
            self.logger.info("Key layer gradient statistics:")
            for name, stats in key_stats.items():
                # Shorten name for readability
                short_name = name.split('.')[-2:] if '.' in name else [name]
                short_name = '.'.join(short_name[-2:])
                
                self.logger.info(
                    f"  {short_name}: mean={stats['mean']:.6f}, "
                    f"std={stats['std']:.6f}, abs_mean={stats['abs_mean']:.6f}, "
                    f"norm={stats['norm']:.4f}, max={stats['max']:.6f}, min={stats['min']:.6f}"
                )
                
                # Log to writer (CometML) - step is already set in _train_epoch
                self.writer.add_scalar(f"grad_stats/{short_name}/mean", stats['mean'])
                self.writer.add_scalar(f"grad_stats/{short_name}/std", stats['std'])
                self.writer.add_scalar(f"grad_stats/{short_name}/abs_mean", stats['abs_mean'])
                self.writer.add_scalar(f"grad_stats/{short_name}/norm", stats['norm'])
                self.writer.add_scalar(f"grad_stats/{short_name}/max", stats['max'])
                self.writer.add_scalar(f"grad_stats/{short_name}/min", stats['min'])
        
        # Log summary statistics
        if zero_grad_ratio > 0.1:
            self.logger.warning(
                f"High ratio of zero gradients: {zero_grad_ratio:.1%}. "
                f"This may indicate vanishing gradients or incorrect loss computation."
            )
        
        # Log histogram of gradients for key layers (if supported)
        for name, stats in key_stats.items():
            if name in grad_stats and not grad_stats[name].get('zero_grad', False):
                try:
                    param = dict(self.model.named_parameters())[name]
                    if param.grad is not None:
                        # Flatten gradient for histogram
                        grad_flat = param.grad.data.flatten().cpu()
                        # Sample if too large (for performance)
                        if grad_flat.numel() > 10000:
                            grad_flat = grad_flat[::grad_flat.numel() // 10000]
                        
                        short_name = '.'.join(name.split('.')[-2:])
                        self.writer.add_histogram(f"grad_hist/{short_name}", grad_flat)
                except Exception as e:
                    self.logger.debug(f"Could not log gradient histogram for {name}: {e}")
