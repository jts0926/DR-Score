import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from loss_func import NegativeLogLikelihood

class BaseModel(pl.LightningModule):
    def __init__(self, num_classes=1, loss='deepsurv', lr=0.01, config=None):
        super(BaseModel, self).__init__()

        self.model = None  # Initialize this as your main model

        if loss == 'deepsurv':
            self.loss_fc = NegativeLogLikelihood(L2_reg=1e-4)  # Note the instantiation here
        else:
            # Add any other loss functions as required
            raise ValueError(f"Unknown loss function: {loss}")

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, event_status, time = batch  # Adjust the unpacking based on your dataset's __getitem__
        outputs = self(data)
        loss = self.loss_fc(outputs, time, event_status, self.model)  # Adjusted the loss calculation here
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, event_status, time = batch  # Adjust the unpacking based on your dataset's __getitem__
        outputs = self(data)
        loss = self.loss_fc(outputs, time, event_status, self.model)  # Adjusted the loss calculation here
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=1e-6,
                                      amsgrad=True)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           0.999,
                                                           last_epoch=-1,
                                                           verbose=False)
        return [optimizer], [scheduler]

