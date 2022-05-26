#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from email.policy import default
import os
import h5py
import random
import re
import scipy.io
import numpy as np
import itertools

from typing import Any, Tuple
from dataclasses import dataclass

Array = Any
os.environ["PYTHONIOENCODING"] = "UTF-8"


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--in_path", type=str,
        help="path/to/design/matrix")
    aa("--out_path", type=str,
        help="path/to/similarity/judgments")
    aa("--n_samples", type=int, 
        help="number of similarity judgements")
    aa("--k", type=int, default=3, choices=[2, 3],
        help='whether to sample pairs or triplets')
    aa("--rnd_seed", type=int, default=42, 
        help="random seed")
    args = parser.parse_args()
    return args


@dataclass
class Sampler:
    in_path: str
    out_path: str
    n_samples: int
    rnd_seed: int
    k: int = 3
    train_frac: float = 0.8

    def __post_init__(self) -> None:
        if not re.search(r"(mat|txt|csv|npy|hdf5)$", self.in_path):
            raise Exception(
                "\nCannot tripletize input data other than .mat, .txt, .csv, .npy, or .hdf5 formats\n"
            )
        if not os.path.exists(self.out_path):
            print(f"\n....Creating output directory: {self.out_path}\n")
            os.makedirs(self.out_path)

        random.seed(self.rnd_seed)
        np.random.seed(self.rnd_seed)

    def load_domain(self, in_path: str) -> Array:
        if re.search(r"mat$", in_path):
            X = np.vstack(
                [
                    v
                    for v in scipy.io.loadmat(in_path).values()
                    if isinstance(v, Array) and v.dtype == np.float
                ]
            )
        elif re.search(r"txt$", in_path):
            X = np.loadtxt(in_path)
        elif re.search(r"csv$", in_path):
            X = np.loadtxt(in_path, delimiter=",")
        elif re.search(r"npy$", in_path):
            X = np.load(in_path)
        elif re.search(r"hdf5$", in_path):
            with h5py.File(in_path, "r") as f:
                X = list(f.values())[0][:]
        else:
            raise Exception("\nInput data does not seem to be in the right format\n")
        X = self.remove_nans_(X)
        return X

    @staticmethod
    def remove_nans_(X: Array) -> Array:
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        return X[~nan_indices]

    @staticmethod
    def softmax(z: Array) -> Array:
        return np.exp(z) / np.sum(np.exp(z))

    def get_choice(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        # TODO: change temperature value (i.e., beta param) because 
        # softmax yields NaNs if dot products are too large
        # probas = self.softmax(sims)
        # positive = combs[np.argmax(probas)]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    @staticmethod
    def random_choice(n_samples: int, combs: Array):
        return combs[np.random.choice(np.arange(combs.shape[0]), size=n_samples, replace=False)]

    @staticmethod
    def get_combinations(M, k):
        return np.array(list(itertools.combinations(range(M), k)))

    @staticmethod
    def cosine_matrix(X: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
        """Compute cosine-similarity matrix."""
        num = X @ X.T
        # compute vector l2-norm across rows
        l2_norms = np.linalg.norm(X, axis=1)
        denom = np.outer(l2_norms, l2_norms)
        cos_mat = (num / denom).clip(min=a_min, max=a_max)
        return cos_mat

    def sample_pairs(self) -> Array:
        X = self.load_domain(self.in_path)
        M = X.shape[0]
        S = self.cosine_matrix(X)
        combs = self.get_combinations(M, self.k)
        random_sample = self.random_choice(self.n_samples, combs)
        return S, random_sample

    def sample_similarity_judgements(self) -> Array:
        """Create similarity judgements."""
        X = self.load_domain(self.in_path)
        M = X.shape[0]
        S = X @ X.T
        similarity_judgements = np.zeros((self.n_samples, self.k), dtype=int)
        combs = self.get_combinations(M, self.k)
        random_sample = self.random_choice(self.n_samples, combs)
        for i, triplet in enumerate(random_sample):
            choice = self.get_choice(S, triplet)
            similarity_judgements[i] = choice
        return similarity_judgements

    def create_train_test_split(
        self, similarity_judgements: Array
    ) -> Tuple[Array, Array]:
        """Split triplet data into train and test splits."""
        N = similarity_judgements.shape[0]
        rnd_perm = np.random.permutation(N)
        train_split = similarity_judgements[rnd_perm[: int(len(rnd_perm) * self.train_frac)]]
        test_split = similarity_judgements[rnd_perm[int(len(rnd_perm) * self.train_frac):]]
        return train_split, test_split

    def save_similarity_judgements(self, similarity_judgements: Array) -> None:
        train_split, test_split = self.create_train_test_split(similarity_judgements)
        with open(os.path.join(self.out_path, "train_90.npy"), "wb") as train_file:
            np.save(train_file, train_split)
        with open(os.path.join(self.out_path, "test_10.npy"), "wb") as test_file:
            np.save(test_file, test_split)


if __name__ == "__main__":
    # parse arguments
    args = parseargs()

    tripletizer = Sampler(
        in_path=args.in_path,
        out_path=args.out_path,
        n_samples=args.n_samples,
        k=args.k,
        rnd_seed=args.rnd_seed,
    )
    if args.k == 3:
        similarity_judgements = tripletizer.sample_similarity_judgements()
        tripletizer.save_similarity_judgements(similarity_judgements)
    else:
        S, random_sample = tripletizer.sample_pairs()
        tripletizer.save_similarity_judgements(random_sample)
        with open(os.path.join(args.out_path, "similarity_matrix.npy"), "wb") as f:
            np.save(f, S)
