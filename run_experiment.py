import torch
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tsl import logger
from datetime import datetime
from tsl.engines import Predictor
from tsl.experiment import Experiment
from tsl.metrics import torch as torch_metrics
from lib.utils import (get_dataset, get_datamodule, get_model_class, get_logger, do_test, ROOT_PATH,
                       future_conditioning_by_name)


def run_forecast(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=True)

    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset.name)

    # Future conditioning info according to the model
    use_fut_cov = future_conditioning_by_name(cfg.model.name)

    dm, torch_dataset = get_datamodule(cfg, dataset, use_fut_cov)

    logger.info(dm)
    ########################################
    # predictor                            #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    # Get dimensions (number) of the covariates and exogenous variables for defining the model
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon,
                        exog_size=torch_dataset.input_map.u.shape[-1] if cfg.use_datetime else 0,
                        past_cov_size=torch_dataset.input_map.past_cov.shape[-1] if cfg.use_past_cov else 0,
                        fut_cov_size=torch_dataset.input_map.fut_cov.shape[-1] if use_fut_cov else 0,
                        )

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)
    logger.info(
        f'GENERAL INFO:\nmodel_kwargs: {model_kwargs}\nmodel_cls: {model_cls}\nmodel_hparam: {cfg.model.hparams}')

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        min_delta=0.05,
                                        patience=cfg.patience,
                                        verbose=True,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        auto_insert_metric_name=True,
        save_last=True,
        verbose=True,
        mode='min',
    )

    exp_logger = get_logger(cfg)

    trainer = Trainer(
        max_epochs=cfg.epochs,
        precision= '16-mixed', # 'bf16-mixed'
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        devices=cfg.devices,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=5,
    )

    trainer.fit(predictor, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()

    # Cycle over different dataloaders [aka over different delayed horizon window]
    # NB. the first dataloader is the backtesting one
    delays = cfg.delays  # i.e. [0, 48, 24, 0] aka [backtesting, 3rd, 2nd, 1st]
    stride = 24
    if len(delays) > 1: # create the datamodules for the different delays
        dm48, _ = get_datamodule(cfg, dataset, use_fut_cov, delay=delays[1])
        dm24, _ = get_datamodule(cfg, dataset, use_fut_cov, delay=delays[2])
        backtesting_dm, _ = get_datamodule(cfg, dataset, use_fut_cov, delay=delays[0], stride=stride)  # dm0_s24
        datamodules = [backtesting_dm, dm48, dm24, dm]
    else:
        datamodules = [dm]
    for i, (dm, delay) in enumerate(zip(datamodules, delays)):
        backtesting = True if i == 0 and delays[0] == 0 else False
        if backtesting:
            logger.info(f'\n==> Backtesting on delay {delay} with stride={stride} (real scenario) <==')
        do_test(predictor=predictor,
                trainer=trainer,
                datamodule=dm,
                delay=delay,
                cfg=cfg,
                save_predictions=True,
                backtesting=backtesting,
        )


if __name__ == '__main__':
    logger.info('\n\n\n==> Start <==\n\n\n')

    start_time = datetime.now()
    exp = Experiment(run_fn=run_forecast,
                     config_path=ROOT_PATH,
                     config_name='default')
    exp.run()

    end_time = datetime.now()
    logger.info(f'End of script\n\nElapsed time: {end_time - start_time}')
    logger.info('\n\n\n==> End <==\n\n\n')
