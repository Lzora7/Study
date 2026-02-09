import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import GANTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Training script for iSTFTNet vocoder with discriminator (GAN training).
    
    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder=None, device=device)

    # SETUP

    # generator
    generator = instantiate(config.model).to(device)
    logger.info("Generator:")
    logger.info(generator)

    # discriminator
    discriminator = instantiate(config.discriminator).to(device)
    logger.info("Discriminator:")
    logger.info(discriminator)

    # loss
    criterion = instantiate(config.loss_function).to(device)
    logger.info(f"Loss function: {type(criterion).__name__}")

    # metrics (may be empty for vocoder, no text_encoder needed)
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # optimizers (gen, dis)
    generator_params = filter(lambda p: p.requires_grad, generator.parameters())
    discriminator_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    
    optimizer_g = instantiate(config.optimizer_g, params=generator_params)
    optimizer_d = instantiate(config.optimizer_d, params=discriminator_params)
    
    logger.info(f"Generator optimizer: {type(optimizer_g).__name__}")
    logger.info(f"Discriminator optimizer: {type(optimizer_d).__name__}")

    # (opt) learning rate schedulers
    lr_scheduler_g = instantiate(config.get("lr_scheduler_g"), optimizer=optimizer_g) if config.get("lr_scheduler_g") else None
    lr_scheduler_d = instantiate(config.get("lr_scheduler_d"), optimizer=optimizer_d) if config.get("lr_scheduler_d") else None

    # epoch_len
    epoch_len = config.trainer.get("epoch_len")

    trainer = GANTrainer(
        model=generator,
        discriminator=discriminator,
        criterion=criterion,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        text_encoder=None,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
