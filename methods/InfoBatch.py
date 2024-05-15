from .SelectionMethod import SelectionMethod
import torch
import numpy as np
import time

class InfoBatch(SelectionMethod):
    method_name = 'InfoBatch'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.loss_percentile = config['loss_percentile']
        self.ratio = config['ratio']
        self.keep_ratio = (self.ratio - 1 + self.loss_percentile) / self.loss_percentile
        self.scores = torch.ones(self.num_train_samples) * 3
        self.weights = torch.ones(self.num_train_samples)

    def before_epoch(self, epoch):
        threshhold = np.percentile(self.scores.numpy(), self.loss_percentile*100)
        well_learned_mask = (self.scores < threshhold).numpy()
        if well_learned_mask.sum() == 0:
            # random generate
            well_learned_mask = np.random.choice([True, False], self.num_train_samples, p=[self.loss_percentile, 1-self.loss_percentile])
        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        real_loss_percentile = len(well_learned_indices) / self.num_train_samples
        self.logger.info(f'Epoch: {epoch}, Max loss: {self.scores.max().item()}, Min loss: {self.scores.min().item()}, Mean loss: {self.scores.mean().item()}, threshhold: {threshhold}, Real loss percentile: {real_loss_percentile}')

        self.keep_ratio = (self.ratio - 1 + real_loss_percentile) / real_loss_percentile
        selected_indices = np.random.choice(well_learned_indices, int(
            self.keep_ratio * len(well_learned_indices)), replace=False)
        self.weights[:] = 1
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        np.random.shuffle(remained_indices)
        self.logger.info(f'Epoch: {epoch}, Selected {len(remained_indices)} samples, Total {self.num_train_samples} samples')
        return remained_indices
    
    def train(self, epoch, list_of_train_idx):
        # train for one epoch
        self.model.train()
        self.logger.info('Epoch: [{} | {}] LR: {}'.format(epoch, self.epochs, self.optimizer.param_groups[0]['lr']))

        # data loader
        # sub_train_dset = torch.utils.data.Subset(self.train_dset, list_of_train_idx)
        list_of_train_idx = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(list_of_train_idx, batch_size=self.batch_size,
                                                      drop_last=False)
        # list_of_train_idx = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.train_dset, num_workers=self.num_data_workers, pin_memory=True, batch_sampler=batch_sampler)
        total_batch = len(train_loader)
        epoch_begin_time = time.time()
        # train
        for i, datas in enumerate(train_loader):
            inputs = datas['input'].cuda()
            targets = datas['target'].cuda()
            indexes = datas['index']
            inputs, targets, indexes = self.before_batch(i, inputs, targets, indexes, epoch)
            outputs, features = self.model(inputs, self.need_features) if self.need_features else (self.model(inputs, False), None)
            # loss = self.criterion(outputs, targets)
            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            # self.while_update(outputs, loss, targets, epoch, features, indexes, batch_idx=i, batch_size=self.batch_size)
            weights = self.weights[indexes].to(loss.device)
            loss_val = loss.detach().clone()
            self.scores[indexes.numpy()] = loss_val.cpu()
            loss = (loss * weights).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.after_batch(i,inputs, targets, indexes,outputs.detach())
            if i % self.config['logger_opt']['print_iter'] == 0:
                # train acc
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()
                train_acc = correct / total
                self.logger.info(f'Epoch: {epoch}/{self.training_opt["num_epochs"]}, Iter: {i}/{total_batch}, global_step: {self.total_step+i}, Loss: {loss.item():.4f}, Train acc: {train_acc:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                    
        self.scheduler.step()
        self.total_step = self.total_step + total_batch
        # test
        now = time.time()
        self.logger.wandb_log({'loss': loss.item(), 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], self.training_opt['loss_type']: loss.item()})
        val_acc = self.test()
        self.logger.wandb_log({'val_acc': val_acc, 'epoch': epoch, 'total_time': now - self.run_begin_time, 'total_step': self.total_step, 'time_epoch': now - epoch_begin_time, 'best_val_acc': max(self.best_acc, val_acc)})
        # self.logger.info(f'=====> Best val acc: {max(self.best_acc, val_acc):.4f}, Current val acc: {val_acc:.4f}')
        self.logger.info(f'=====> Time: {now - self.run_begin_time:.4f} s, Time this epoch: {now - epoch_begin_time:.4f} s, Total step: {self.total_step}')
            # save model
        self.logger.info('=====> Save model')
        is_best = False
        if val_acc > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = val_acc
            is_best = True
        checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_acc': self.best_acc,
                'best_epoch': self.best_epoch
            }
        self.save_checkpoint(checkpoint, is_best)
        self.logger.info(f'=====> Epoch: {epoch}/{self.epochs}, Best val acc: {self.best_acc:.4f}, Current val acc: {val_acc:.4f}')
        self.logger.info(f'=====> Best epoch: {self.best_epoch}')
