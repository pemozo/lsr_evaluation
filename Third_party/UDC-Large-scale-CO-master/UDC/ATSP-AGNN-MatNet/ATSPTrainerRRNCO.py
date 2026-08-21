import os
import time

import torch

from ATSPTrainerPartition import ATSPTrainerPartition
from RRNCOATSPEnv import RRNCOATSPEnv
from RRNCOProblemSampler import RRNCOProblemSampler
from utils.utils import util_print_log_array


class ATSPTrainerRRNCO(ATSPTrainerPartition):
    """Fine-tune both UDC ATSP models on on-the-fly RRNCO instances."""

    def __init__(
            self,
            env_params,
            model_params,
            model_p_params,
            optimizer_params,
            trainer_params,
            rrnco_params,
            checkpoint_params):
        super().__init__(
            env_params=env_params,
            model_params=model_params,
            model_p_params=model_p_params,
            optimizer_params=optimizer_params,
            trainer_params=trainer_params,
        )

        self.device = next(self.model_t.parameters()).device
        self.rrnco_params = rrnco_params
        self.checkpoint_params = checkpoint_params
        self.train_sampler = RRNCOProblemSampler(
            data_dir=rrnco_params['data_dir'],
            seed=rrnco_params['train_seed'],
            cache_size=rrnco_params['cache_size'],
        )
        self.validation_sampler = RRNCOProblemSampler(
            data_dir=rrnco_params['data_dir'],
            seed=rrnco_params['validation_seed'],
            cache_size=rrnco_params['cache_size'],
        )
        self.env = RRNCOATSPEnv(
            problem_sampler=self.train_sampler,
            device=self.device,
            sampler_seed=rrnco_params['size_seed'],
            **env_params
        )

        self._load_source_weights()
        self._prepare_checkpoint_targets()

    def _load_source_weights(self):
        t_file = self._resolve_required_file(self.checkpoint_params['source_t_file'])
        p_file = self._resolve_required_file(self.checkpoint_params['source_p_file'])
        t_checkpoint = torch.load(t_file, map_location=self.device)
        p_checkpoint = torch.load(p_file, map_location=self.device)
        self.model_t.load_state_dict(t_checkpoint['model_state_dict'])
        self.model_p.load_state_dict(p_checkpoint['model_state_dict'])
        self.source_t_epoch = t_checkpoint.get('epoch')
        self.source_p_epoch = p_checkpoint.get('epoch')
        self.source_t_file = t_file
        self.source_p_file = p_file
        self.start_epoch = 1
        self.logger.info('Loaded source TSP weights from %s (epoch %s)', t_file, self.source_t_epoch)
        self.logger.info('Loaded source partition weights from %s (epoch %s)', p_file, self.source_p_epoch)
        self.logger.info('Using fresh optimizers for RRNCO fine-tuning')

    def _prepare_checkpoint_targets(self):
        output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(
            self.checkpoint_params['output_dir']
        )))
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_kind = self.checkpoint_params['kind']
        self.output_t_file = os.path.join(
            output_dir, 'checkpoint-tsp-rrnco-{}.pt'.format(checkpoint_kind)
        )
        self.output_p_file = os.path.join(
            output_dir, 'checkpoint-partition-rrnco-{}.pt'.format(checkpoint_kind)
        )
        existing = [path for path in (self.output_t_file, self.output_p_file) if os.path.exists(path)]
        if existing:
            raise FileExistsError('Refusing to overwrite existing checkpoint: {}'.format(existing[0]))

    @staticmethod
    def _resolve_required_file(path):
        resolved = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if not os.path.isfile(resolved):
            raise FileNotFoundError('Source checkpoint not found: {}'.format(resolved))
        return resolved

    def run(self):
        validation_episodes = self.trainer_params['validation_test_episodes']
        data_250 = self.validation_sampler.sample(
            batch_size=validation_episodes,
            node_count=250,
            split='test',
        ).to(self.device)
        data_500 = self.validation_sampler.sample(
            batch_size=validation_episodes,
            node_count=500,
            split='test',
        ).to(self.device)

        total_epochs = self.trainer_params['epochs']
        validation_interval = self.trainer_params['validation_interval']
        training_started = time.perf_counter()
        for epoch in range(1, total_epochs + 1):
            self.logger.info('=================================================================')
            active_cities = self.train_sampler.start_epoch()
            self.logger.info('RRNCO training cities for epoch %d: %s', epoch, ', '.join(active_cities))
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            if epoch % validation_interval == 0 or epoch == total_epochs:
                self.validation(250, data_250)
                self.validation(500, data_500)

            self.scheduler_p.step()
            self.scheduler_t.step()
            elapsed_seconds = time.perf_counter() - training_started
            mean_epoch_seconds = elapsed_seconds / epoch
            remaining_seconds = mean_epoch_seconds * (total_epochs - epoch)
            self.logger.info(
                'Epoch %3d/%3d: elapsed %.2fh, estimated remaining %.2fh',
                epoch,
                total_epochs,
                elapsed_seconds / 3600.0,
                remaining_seconds / 3600.0,
            )

        total_runtime = time.perf_counter() - training_started
        self._save_final_checkpoints(total_epochs, total_runtime)
        self.logger.info(' *** RRNCO Fine-tuning Done *** ')
        self.logger.info('TSP checkpoint: %s', self.output_t_file)
        self.logger.info('Partition checkpoint: %s', self.output_p_file)
        util_print_log_array(self.logger, self.result_log)

    def _save_final_checkpoints(self, epoch, total_runtime):
        common_metadata = {
            'epoch': epoch,
            'source_t_file': self.source_t_file,
            'source_p_file': self.source_p_file,
            'source_t_epoch': self.source_t_epoch,
            'source_p_epoch': self.source_p_epoch,
            'fine_tune_kind': self.checkpoint_params['kind'],
            'rrnco_data_dir': os.path.abspath(self.rrnco_params['data_dir']),
            'problem_sizes': self.env.problem_sizes,
            'total_runtime_seconds': total_runtime,
        }
        checkpoint_dict_t = dict(common_metadata)
        checkpoint_dict_t.update({
            'model_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer_t.state_dict(),
            'scheduler_state_dict': self.scheduler_t.state_dict(),
            'result_log': self.result_log.get_raw_data(),
        })
        checkpoint_dict_p = dict(common_metadata)
        checkpoint_dict_p.update({
            'model_state_dict': self.model_p.state_dict(),
            'optimizer_state_dict': self.optimizer_p.state_dict(),
            'scheduler_state_dict': self.scheduler_p.state_dict(),
            'result_log': self.result_log.get_raw_data(),
        })
        torch.save(checkpoint_dict_t, self.output_t_file)
        torch.save(checkpoint_dict_p, self.output_p_file)
