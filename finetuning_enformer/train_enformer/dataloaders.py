# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, DistributedSampler
from .datasets import RealRegressionDataset, RealPairDataset, VirtualPairDataset #, VirtualRegressionDataset
from .utils import seed_worker

# def create_virtual_regression_loader(genes_file, vector_dir, pseudo_label_path,
#                           batch_size=1, training=True, pairs_per_epoch=None):
#     dataset = VirtualRegressionDataset(genes_file, vector_dir, pseudo_label_path,
#                                  n_virtual_samples=1000, pairs_per_epoch=pairs_per_epoch)
#     if training:
#         sampler = DistributedSampler(dataset, shuffle=True)
#         return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
#                           num_workers=4, prefetch_factor=2, drop_last=True,
#                           worker_init_fn=seed_worker)
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                           num_workers=4, prefetch_factor=2, worker_init_fn=seed_worker)

def create_virtual_pair_loader(genes_file, vector_dir, pseudo_label_path,
                          batch_size=1, training=True, pairs_per_epoch=None):
    dataset = VirtualPairDataset(genes_file, vector_dir, pseudo_label_path,
                                 n_virtual_samples=1000, pairs_per_epoch=pairs_per_epoch)
    if training:
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=4, prefetch_factor=2, drop_last=True,
                          worker_init_fn=seed_worker)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, prefetch_factor=2, worker_init_fn=seed_worker)


def create_real_pair_loader(genes_file, samples_file, data_dir,
                            batch_size=1, training=True, pairs_per_epoch=None):
    dataset = RealPairDataset(genes_file, samples_file, data_dir, pairs_per_epoch=pairs_per_epoch)
    if training:
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=4, prefetch_factor=2, drop_last=True,
                          worker_init_fn=seed_worker)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, prefetch_factor=2, worker_init_fn=seed_worker)


def create_real_regression_loader(genes_file, samples_file, data_dir,
                       batch_size=1, training=False):
    dataset = RealRegressionDataset(genes_file, samples_file, data_dir)
    if training:
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=4, prefetch_factor=2, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, prefetch_factor=2)