import pickle
import torch
from dataset.data_module_new_full import \
	MFSEDataModule
from src.mfse_model import Setting, MFSE

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

MAX_EPOCH = 2

# set training device
device = "0" if torch.cuda.is_available() else "cpu"
#device = "cpu"

# setup data module
mfse_data_module = MFSEDataModule(sp_rate=0.9, fold=0, n_splits=10)
mfse_data_module.prepare_data()

# initialize model
settings = Setting(sp_rate=0.9, lr=0.01, drug_target_dim=20, n_embed=283,
				   n_hid1=64, n_hid2=20, num_base=32)
model = MFSE(settings, mfse_data_module, MAX_EPOCH, device)
logger = TensorBoardLogger('./lightning_logs')

trainer = Trainer(max_epochs = MAX_EPOCH, log_every_n_steps=149, logger=logger, gpus=1)
#trainer = Trainer(max_epochs = MAX_EPOCH, log_every_n_steps=149, logger=logger)
trainer.fit(model, mfse_data_module)
print(trainer.logged_metrics)

# save results
with open(f'./saved_records/record_{trainer.logger.version}.pkl', 'wb') as handle:
	pickle.dump(model.record, handle)
with open(f'./saved_records/record_per_se_degree_{trainer.logger.version}.pkl', 'wb') as handle2:
	pickle.dump(model.record_per_se_degree, handle2)
torch.save(model, f'./saved_model/model_{trainer.logger.version}.pt')