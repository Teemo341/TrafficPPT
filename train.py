import json
import os
import argparse
import torch
import numpy as np
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import datetime

from model.model_mae import no_diffusion_model_cross_attention_parallel as no_diffusion_model
from dataloader.dataloader import  traj_dataloader
from global_settings import global_data_root


class pl_wrapper(pl.LightningModule):
    def __init__(self, model, optimizer_class, lr, max_epochs, sch_class, lr_decay, special_mask_value, use_condition=True, condition_observable=False, use_adj_table=True):
        super().__init__()
        self.model = model
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.max_epochs = max_epochs
        self.sch_class = sch_class
        self.lr_decay = lr_decay
        self.special_mask_value = special_mask_value
        self.use_condition = use_condition
        self.condition_observable = condition_observable
        self.use_adj_table = use_adj_table
        self.save_hyperparameters(ignore=['model'])

    def load_model(self, model_ckpt_path):
        checkpoint = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def configure_optimizers(self):
        if self.optimizer_class == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-15)
        elif self.optimizer_class == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-15)
        elif self.optimizer_class == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError('Unsupported optimizer class')

        if self.lr_decay is None or self.lr_decay == 0:
            return {'optimizer': optimizer}
        else:
            assert self.lr_decay <1 and self.lr_decay >0, 'lr_decay should be between 0 and 1'
            if self.sch_class == 'StepLR':
                gamma = 0.5
                freq = torch.log10(torch.tensor(self.lr_decay))/torch.log10(torch.tensor(gamma))
                save_iters = int(self.max_epochs/freq)
                lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_iters, gamma=gamma)
            elif self.sch_class == 'CosineAnnealingLR':
                lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.lr_decay, last_epoch=-1)
            else:
                raise ValueError('Unsupported scheduler class')

            return {'optimizer': optimizer, 'lr_scheduler': lr_sched}

    def forward(self, batch):
        current_time = time.time()
        condition, time_step, special_mask, adj_table = batch
        # trajectory: [B x N x T], time_step: [B x N], special_mask: [B x N x T], adj_table: [B x 1 x V x 4 x 2]
        condition, time_step, special_mask, adj_table = condition.to(self.device), time_step.to(self.device), special_mask.to(self.device), adj_table.to(self.device)

        # random choice a traj as input, the rest as condition
        shuffled_indices = torch.randperm(condition.size(1))
        condition = condition[:,shuffled_indices,:]
        time_step = time_step[:,shuffled_indices]
        special_mask = special_mask[:,shuffled_indices,:]

        # get y, filter trajecotry into condition and get x
        y = condition[:,0,:] # [B x T]
        y = y.long()
        condition_ = self.filter_condition(condition) # remove unboservable nodes

        x = condition_[:,0,:] # [B x T]
        if self.use_condition:
            if self.condition_observable:
                condition = condition[:,1:,:] # [B x N-1 x T]
            else:
                condition = condition_[:,1:,:] # [B x N-1 x T]
        else:
            condition = None

        if self.use_adj_table:
            adj_table = adj_table[:,shuffled_indices[0],:,:,:] # [B x V x 4 x 2]
        else:
            adj_table = None
        
        special_mask = special_mask[:,0,:]
        special_mask_ = (special_mask+self.special_mask_value).clamp(0,1).float()

        preprocess_data_time = time.time()-current_time
        current_time = time.time()

        logits, loss = self.model(x, condition, adj_table, y, time_step, None, special_mask_)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        acc_inner = ((torch.argmax(logits, dim=-1) == y).float()*special_mask).sum()/special_mask.sum()
        real_true_rate = 0
        for j in range(y.shape[0]):
            if (torch.argmax(logits[j], dim = -1) == y[j]).all():
                real_true_rate += 1/y.shape[0]

        forward_time = time.time()-current_time
        return logits, loss, acc, acc_inner, real_true_rate, preprocess_data_time, forward_time

    def filter_condition(self, condition):
        '''a wrapper function to filter condition, replace with dataloader's filter_condition'''
        raise NotImplementedError('Please implement filter_condition function in pl_wrapper class')
    
    def training_step(self, batch, batch_idx):
        logits, loss, acc, acc_inner, real_true_rate, preprocess_data_time, forward_time = self.forward(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc_inner', acc_inner, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/real_true_rate', real_true_rate, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('time/train/preprocess_data', preprocess_data_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('time/train/forward', forward_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            logits, loss, acc, acc_inner, real_true_rate, preprocess_data_time, forward_time = self.forward(batch)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc_inner', acc_inner, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/real_true_rate', real_true_rate, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('time/val/preprocess_data', preprocess_data_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('time/val/forward', forward_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss


class TimeCallback(Callback):
    def __init__(self, log_every_n=1):
        self.prev_batch_end_time = None
        self.log_every_n = log_every_n
        self.batch_idx = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.prev_batch_end_time = time.perf_counter()
        self.batch_idx = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        # 近似的数据加载等待时间 = 本 batch 开始时间 - 上一个 batch 结束时间
        if self.prev_batch_end_time is not None and (self.batch_idx % self.log_every_n) == 0:
            data_wait = now - self.prev_batch_end_time
            pl_module.log("time/train/load_data", data_wait, on_step=True, on_epoch=True, prog_bar=False)
        # 标记前向-反向-优化的开始
        self.compute_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (self.batch_idx % self.log_every_n) == 0:
            compute_time = time.perf_counter() - self.compute_start
            pl_module.log("time/train/total_batch", compute_time, on_step=True, on_epoch=True, prog_bar=False)

        self.prev_batch_end_time = time.perf_counter()
        self.batch_idx += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.prev_batch_end_time = time.perf_counter()
        self.batch_idx = 0
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        # 近似的数据加载等待时间 = 本 batch 开始时间 - 上一个 batch 结束时间
        if self.prev_batch_end_time is not None and (self.batch_idx % self.log_every_n) == 0:
            data_wait = now - self.prev_batch_end_time
            pl_module.log("time/val/load_data", data_wait, on_step=True, on_epoch=True, prog_bar=False)
        # 标记前向-反向-优化的开始
        self.compute_start = time.perf_counter()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (self.batch_idx % self.log_every_n) == 0:
            compute_time = time.perf_counter() - self.compute_start
            pl_module.log("time/val/total_batch", compute_time, on_step=True, on_epoch=True, prog_bar=False)

        self.prev_batch_end_time = time.perf_counter()
        self.batch_idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # about dataset
    parser.add_argument('--city', type=str, default='boston')
    parser.add_argument('--data_root', type=str, default=global_data_root)
    parser.add_argument('--data_type', type=str, default='simple_simulator')
    parser.add_argument('--data_num', type=int, default=500000)
    parser.add_argument('--test_data_num', type=int, default=2000)
    parser.add_argument('--history_num', type=int, default=5, help='on each weighted graph, give how many condition?, same as random_sample_num in dataset')
    parser.add_argument('--weight_quantization_scale', type=int, default=None)
    parser.add_argument('--max_connection', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=50, help='The max length of all trajectories')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--store', type=bool, default=True)

    # about model
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_condition', type=bool, default=True)
    parser.add_argument('--condition_observable', type=bool, default=False)
    parser.add_argument('--use_adj_table', type=bool, default=True)
    parser.add_argument('--adj_type', type=str, default='bv1h', help='bveh, bv1h, b11h, bvkh')
    parser.add_argument('--use_timestep', type=bool, default=False)

    # about training
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--special_mask_value', type=float, default=0.0001)
    parser.add_argument('--observe_ratio', type=float, default=0.5)
    parser.add_argument('--reset_observation', type=bool, default=False)
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Double precision (64), full precision (32), 16bit mixed precision (16, 16-mixed) or bfloat16 mixed precision (bf16, bf16-mixed).')

    # about record and resume training
    parser.add_argument('--logdir', type=str, default='./logs', help='directory to save pl logs')
    parser.add_argument('--resume_dir', type=str, default=None, help='directory to resume training from pl logs')
    parser.add_argument('--pretrain_model_dir', type=str, default=None)

    args = parser.parse_args()
    if args.weight_quantization_scale == 0:
        args.weight_quantization_scale = None
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    namenow = args.city + '_' + now
    args.logdir = os.path.join(args.logdir, namenow)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir,'args.json'), 'wb') as f:
        f.write(json.dumps(vars(args), indent=4).encode('utf-8'))
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")

    pl.seed_everything(args.seed)

    # Load data ------------------------------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    dataloader = traj_dataloader(city=args.city, data_root=args.data_root, data_type=args.data_type, data_num=args.data_num, test_data_num = args.test_data_num, history_num=args.history_num, block_size=args.block_size, weight_quantization_scale=args.weight_quantization_scale, max_connection=args.max_connection, batch_size=args.batch_size, num_workers=8, seed=args.seed, store=args.store)
    dataloader.randomize_condition(args.observe_ratio) # random initialize condition
    vocab_size = dataloader.vocab_size
    print(f'{args.city} has {vocab_size -1} nodes, add 0 for sepcial token, now vocab size is {vocab_size}')
    elapsed_time = time.time() - start_time
    print(f'Data loaded in {elapsed_time//60} minutes {elapsed_time%60:.2f} seconds')

    # Load model ------------------------------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    model= no_diffusion_model(vocab_size=vocab_size, n_embd=args.n_embd, n_hidden=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, block_size=args.block_size, dropout=args.dropout, weight_quantization_scale=args.weight_quantization_scale, use_condition=args.use_condition,use_adj_table=args.use_adj_table, use_timestep=args.use_timestep, adj_type = args.adj_type)
    pl_model = pl_wrapper(model, optimizer_class='AdamW', lr=args.learning_rate, max_epochs=args.max_epochs, sch_class='CosineAnnealingLR', lr_decay=args.lr_decay, special_mask_value=args.special_mask_value, use_condition=args.use_condition, condition_observable=args.condition_observable, use_adj_table=args.use_adj_table)
    if args.pretrain_model_dir is not None:
        assert os.path.exists(args.pretrain_model_dir), f'pretrain model path {args.pretrain_model_dir} does not exist'
        print('Loading pre-trained model from', args.pretrain_model_dir)
        pl_model.load_model(args.pretrain_model_dir)
    pl_model.filter_condition = dataloader.filter_condition  # assign filter_condition function from dataloader to pl_model
    elapsed_time = time.time() - start_time
    print(f'Model loaded in {elapsed_time//60} minutes {elapsed_time%60:.2f} seconds')

    # callbacks ------------------------------------------------------------------------------------------------------------------------------------------
    class MaskObservationCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if args.reset_observation:
                dataloader.randomize_condition(args.observe_ratio)
                pl_module.filter_condition = dataloader.filter_condition
            
    time_callback = TimeCallback(log_every_n=1)
    mask_observation_callback = MaskObservationCallback()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(args.logdir, 'checkpoints'),
        filename='model-{epoch:04d}-{val_loss:.4f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # train

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.logdir,
        accelerator='auto',
        devices='auto',
        precision=args.precision,
        callbacks=[time_callback, mask_observation_callback, checkpoint_callback],
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
    )
    if args.resume_dir is not None:
        print('Resuming training from', args.resume_dir)
    trainer.fit(pl_model, dataloader.train_loader, dataloader.test_loader, ckpt_path=args.resume_dir)

    print('Training finished')