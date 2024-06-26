import sys
import datetime
import argparse

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument(
            '--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument(
            '--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument(
            '--tb-prefix',
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter."
        )

        parser.add_argument(
            'comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.use_mps = torch.backends.mps.is_available()

        if self.use_cuda:
            self.device = torch.device('cuda')
        elif self.use_mps:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices.")
          if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainD1(self):
        train_ds = LunaDataset(
                val_stride=10,
                isValSet_bool=False,
            )
            batch_size = self.cli_args.batch_size
            if self.use_cuda:
                batch_size *= torch.cuda.device_count()

            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
            )

            return train_dl


if __name__ == '__main__':
    LunaTrainingApp().main()
