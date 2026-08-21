import json
import os
from collections import OrderedDict

import numpy as np
import torch


class RRNCOProblemSampler:
    """Sample ATSP distance matrices from RRNCO city data without saving instances."""

    def __init__(
            self,
            data_dir,
            seed=1234,
            split_file='splited_cities_list.json',
            cache_size=10,
            train_cities_per_epoch=10,
            invalid_threshold=1e5):
        self.data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(data_dir)))
        self.rng = np.random.default_rng(seed)
        self.cache_size = cache_size
        self.train_cities_per_epoch = train_cities_per_epoch
        self.invalid_threshold = invalid_threshold
        self._cache = OrderedDict()

        split_path = os.path.join(self.data_dir, split_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError('RRNCO split file not found: {}'.format(split_path))
        with open(split_path, mode='r', encoding='utf-8') as split_handle:
            city_splits = json.load(split_handle)

        self.city_splits = {}
        for split_name in ('train', 'test'):
            cities = city_splits.get(split_name)
            if not cities:
                raise ValueError('RRNCO split "{}" is empty in {}'.format(split_name, split_path))
            missing = [city for city in cities if not os.path.isfile(self._city_path(city))]
            if missing:
                raise FileNotFoundError(
                    'Missing RRNCO city data for split {}: {}'.format(split_name, ', '.join(missing[:5]))
                )
            self.city_splits[split_name] = list(cities)

        self.last_sampled_cities = []
        self.last_sampled_node_count = None
        self.active_train_cities = []

    def start_epoch(self):
        train_cities = self.city_splits['train']
        city_count = min(self.train_cities_per_epoch, len(train_cities))
        selected = self.rng.choice(len(train_cities), size=city_count, replace=False)
        self.active_train_cities = [train_cities[int(index)] for index in selected]
        return list(self.active_train_cities)

    def sample(self, batch_size, node_count, split='train'):
        if batch_size < 1:
            raise ValueError('batch_size must be at least 1')
        if split not in self.city_splits:
            raise ValueError('Unknown RRNCO split: {}'.format(split))

        instances = []
        sampled_cities = []
        if split == 'train':
            if not self.active_train_cities:
                self.start_epoch()
            cities = self.active_train_cities
        else:
            cities = self.city_splits[split]
        for _ in range(batch_size):
            city = cities[int(self.rng.integers(0, len(cities)))]
            city_distance = self._load_city_distance(city)
            if city_distance.shape[0] < node_count:
                raise ValueError(
                    '{} has only {} valid nodes, but {} were requested'.format(
                        city, city_distance.shape[0], node_count
                    )
                )
            indices = self.rng.choice(city_distance.shape[0], size=node_count, replace=False)
            instance = city_distance[np.ix_(indices, indices)].copy()
            np.fill_diagonal(instance, 0.0)
            instances.append(instance)
            sampled_cities.append(city)

        self.last_sampled_cities = sampled_cities
        self.last_sampled_node_count = node_count
        return torch.from_numpy(np.stack(instances, axis=0))

    def _city_path(self, city):
        return os.path.join(self.data_dir, city, '{}_data.npz'.format(city))

    def _load_city_distance(self, city):
        if city in self._cache:
            distance = self._cache.pop(city)
            self._cache[city] = distance
            return distance

        city_path = self._city_path(city)
        try:
            distance = self._read_distance(city_path, allow_pickle=False)
        except ValueError as error:
            if 'Object arrays cannot be loaded' not in str(error):
                raise
            # Bangkok and Singapore in the published RRNCO data are object arrays.
            distance = self._read_distance(city_path, allow_pickle=True)

        distance = np.asarray(distance, dtype=np.float32)
        distance = self._remove_invalid_nodes(distance, city)
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[city] = distance
        return distance

    @staticmethod
    def _read_distance(city_path, allow_pickle):
        with np.load(city_path, allow_pickle=allow_pickle) as city_data:
            if 'distance' not in city_data:
                raise KeyError('Required array "distance" not found in {}'.format(city_path))
            return np.asarray(city_data['distance'])

    def _remove_invalid_nodes(self, distance, city):
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError('RRNCO distance matrix for {} is not square: {}'.format(city, distance.shape))

        invalid = (~np.isfinite(distance)) | (distance > self.invalid_threshold) | (distance < 0)
        active = np.ones(distance.shape[0], dtype=bool)
        while True:
            active_indices = np.flatnonzero(active)
            active_invalid = invalid[np.ix_(active_indices, active_indices)]
            if not active_invalid.any():
                break
            invalid_per_node = active_invalid.sum(axis=0) + active_invalid.sum(axis=1)
            active[active_indices[int(np.argmax(invalid_per_node))]] = False

        valid_indices = np.flatnonzero(active)
        if valid_indices.size < 500:
            raise ValueError('{} has only {} valid nodes after cleaning'.format(city, valid_indices.size))
        return distance[np.ix_(valid_indices, valid_indices)]
