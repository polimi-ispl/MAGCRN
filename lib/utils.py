import os.path
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from tsl.experiment import Experiment
from tsl import logger
from tsl.datasets import TabularDataset
import lib.models as user_models
from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset, TemporalSplitter, SynchMode, AtTimeStepSplitter
from tsl.data.preprocessing import StandardScaler, RobustScaler
from tsl.metrics import torch_metrics
from tsl.nn import models
from omegaconf import DictConfig
from typing import Optional

########################################
# paths                                #
########################################
ROOT_PATH = '/nas/home/agiganti/arianet/magcrn/'

########################################
# name                                 #
########################################
def future_conditioning_by_name(name):
    '''
    Check if the model needs future conditioning.
    Example behaviour:
    name1 = "gatedgnn": False
    name2 = "gwavenet": False
    name3 = "agcrn": False
    name4 = "magcrn": True
    name5 = "other_name": False

    :param name: name of the model
    :return: bool flag for future conditioning
    '''
    return True if name == 'magcrn' else False


########################################
# get                                  #
########################################
def get_model_class(model_str: str):
    model = None
    if model_str == 'gatedgnn':
        model = models.GatedGraphNetworkModel  # (Satorras et al., 2022)
    elif model_str == 'gwavenet':
        model = user_models.GraphWaveNetModelMod  # (Wu et al., IJCAI 2019)
    elif model_str == 'agcrn':
        model = models.AGCRNModel  # (Bai et al., NeurIPS 2020)
    elif model_str == 'magcrn':
        model = user_models.MAGCRN  # (Giganti et al., IGARSS 2024)
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name):
    if dataset_name == 'madrid19':
        dataset = TabularDataset.load_pickle(os.path.join(ROOT_PATH, 'lib', 'dataset', 'Madrid_2019_tsl.pkl'))
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def get_logger(cfg: DictConfig) -> Optional[Logger]:
    if cfg.logger is None:
        return None
    assert 'backend' in cfg.logger, \
        "cfg.logger must have a 'backend' attribute."
    if cfg.logger.backend == 'tensorboard':
        exp_name = f'{cfg.run.name}_{"_".join(cfg.tags)}'
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=exp_name)
    else:
        raise ValueError(f"Logger {cfg.logger.backend} not available.")
    return exp_logger


def get_datamodule(cfg: DictConfig, dataset: TabularDataset, use_fut_cov, delay: Optional[int] = None, stride: Optional[int] = None,
                   window: Optional[int] = None, horizon: Optional[int] = None):
    # —————— Covariates ——————
    covariates = (
        {'past_cov': dataset.past_cov} if  cfg.use_past_cov and not cfg.use_datetime and not use_fut_cov else
        {'past_cov': dataset.past_cov, 'u': dataset.u} if cfg.use_past_cov and cfg.use_datetime and not use_fut_cov else
        {'past_cov': dataset.past_cov, 'fut_cov': dataset.fut_cov, 'u': dataset.u} if cfg.use_past_cov and cfg.use_datetime and use_fut_cov else
        {'past_cov': dataset.past_cov, 'fut_cov': dataset.fut_cov} if cfg.use_past_cov and not cfg.use_datetime and use_fut_cov else
        {'fut_cov': dataset.fut_cov, 'u': dataset.u} if not cfg.use_past_cov and cfg.use_datetime and use_fut_cov else
        {'fut_cov': dataset.fut_cov} if not cfg.use_past_cov and not cfg.use_datetime and use_fut_cov else
        {**dataset.covariates}
    )

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        # connectivity=adj,
        covariates=covariates,
        delay=cfg.delay if delay is None else delay,
        stride=cfg.stride if stride is None else stride,
        window=cfg.window if window is None else window,
        horizon=cfg.horizon if horizon is None else horizon,
    )

    logger.info(
        f'Dataset: {cfg.dataset.name}, '
        f'window: {cfg.window if window is None else window}, '
        f'stride: {cfg.stride if stride is None else stride}, '
        f'horizon: {cfg.horizon if horizon is None else horizon}, '
        f'delay: {cfg.delay if delay is None else delay}')

    # —————— Covariates Synchronization ——————
    # Past covariates <-- WINDOW
    if cfg.use_past_cov:
        # (past_cov)
        torch_dataset.update_covariate('past_cov', value=torch_dataset.past_cov, pattern='tnf', synch_mode=SynchMode.WINDOW)
    if cfg.use_datetime and cfg.use_past_cov:
        # (u+past_cov)
        torch_dataset.add_exogenous(name='global_u', value=torch_dataset.u, synch_mode=SynchMode.WINDOW)
        if ('magcrn' not in cfg.model.name) and cfg.use_past_cov: # needed to keep the original code of the baselines
            # u = u+past_cov
            past_cov_and_u = merge_cov_and_u(torch_dataset.past_cov, torch_dataset.u)
            torch_dataset.update_covariate(name='u', value=past_cov_and_u, pattern='tnf', synch_mode=SynchMode.WINDOW)
            logger.info(f'Past Covariates and U merged together. Final U shape: {past_cov_and_u.shape}.'
                        f'\nIGNORE the WARNING "Arguments ["past_cov"] are filtered out" in the '
                        f'Dataloader Sanity Check.')
    # Future covariates <-- HORIZON
    if use_fut_cov:
        # (fut_cov)
        torch_dataset.update_covariate('fut_cov', value=torch_dataset.fut_cov, pattern='tnf', synch_mode=SynchMode.HORIZON)
    if cfg.use_datetime and use_fut_cov:
        # (u+fut_cov)
        fut_cov_and_u = merge_cov_and_u(torch_dataset.fut_cov, torch_dataset.u)
        torch_dataset.update_covariate('fut_cov', value=fut_cov_and_u, pattern='tnf', synch_mode=SynchMode.HORIZON)

    # —————— Scalers ——————
    transform = ({'target': StandardScaler(axis=(0, 1))})
    if cfg.use_past_cov: transform.update({'past_cov': StandardScaler(axis=(0, 1))})
    if use_fut_cov: transform.update({'fut_cov': StandardScaler(axis=(0, 1))})

    # —————— Splitter ——————
    splitter = TemporalSplitter(val_len=cfg.dataset.splitting.val_len, test_len=cfg.dataset.splitting.test_len)

    # —————— DataModule ——————
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=cfg.batch_size,
        pin_memory=True,
        workers=cfg.workers,
    )
    dm.setup()
    return dm, torch_dataset

########################################
# processing                           #
########################################

def do_test(predictor, trainer, datamodule, cfg, delay=None, save_predictions=False, backtesting=False):
    if delay is None:
        delay = cfg.delay
    logger.info(f' ————————————————— MODEL {cfg.model.name} with DELAY: {delay}  ————————————————— ')

    trainer.test(predictor, datamodule=datamodule, verbose=True)
    output = trainer.predict(predictor, dataloaders=datamodule.test_dataloader())
    output = predictor.collate_prediction_outputs(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res = dict(test_mae=torch_metrics.mae(y_hat, y_true),
               test_rmse=torch_metrics.rmse(y_hat, y_true),
               test_mape=torch_metrics.mape(y_hat, y_true, nan_to_zero=True),
               test_mre=torch_metrics.mre(y_hat, y_true),
               test_r2=torch_metrics.r2(y_hat, y_true))

    if save_predictions:  # save only Test predictions
        y_hat = y_hat.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        filepath_y_hat = os.path.join(cfg.run.dir, f"y_hat_{delay}{'_backtest' if backtesting else ''}.npy")
        filepath_y_true = os.path.join(cfg.run.dir, f"y_true_{delay}{'_backtest' if backtesting else ''}.npy")
        np.save(filepath_y_hat, y_hat)
        np.save(filepath_y_true, y_true)

    output = trainer.predict(predictor, dataloaders=datamodule.val_dataloader())
    output = predictor.collate_prediction_outputs(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('mask', None))
    res.update(
        dict(val_mae=torch_metrics.mae(y_hat, y_true),
             val_rmse=torch_metrics.rmse(y_hat, y_true),
             val_mape=torch_metrics.mape(y_hat, y_true, nan_to_zero=True),
             val_mre=torch_metrics.mre(y_hat, y_true),
             val_r2=torch_metrics.r2(y_hat, y_true)))

    logger.info(res)
    logger.info(f'End of testing net: {cfg.model.name} with delay: {delay}')
    return res

def merge_cov_and_u(cov, u):
    """
    Merge the covariates with the global_u exogenous variable.

    Parameters:
    - cov (np.array): Covariates array. [n_samples, n_nodes, n_features]
    - u (np.array): Global_u exogenous variable array. [n_samples, n_features]

    Returns:
    - np.array: Covariates array with global_u merged. [n_samples, n_nodes, n_features]
    """
    u = np.expand_dims(u, axis=1)
    u_node_level = np.repeat(u, repeats=cov.shape[1], axis=1)
    cov_and_u = np.concatenate([cov, u_node_level], axis=-1)
    return cov_and_u

def encode_weekday(datetime_index):
    """
    Encode the weekday of a given DatetimeIndex using a boolean vector.

    Parameters:
    - datetime_index (pd.DatetimeIndex): DatetimeIndex for which to perform encoding.

    Returns:
    - pd.DataFrame: DataFrame with One-hot encoded vector representing weekday vector.
    """

    # Extract weekday from datetime
    weekday = datetime_index.dayofweek

    # Create boolean vector for the corresponding weekday
    encoded_weekday_vector = pd.DataFrame({
        'is_monday': (weekday == 0),
        'is_tuesday': (weekday == 1),
        'is_wednesday': (weekday == 2),
        'is_thursday': (weekday == 3),
        'is_friday': (weekday == 4),
        'is_saturday': (weekday == 5),
        'is_sunday': (weekday == 6),
    }).values.astype(bool)

    return encoded_weekday_vector
