# coding: utf-8

from .centaur import Centaur


def create_model(hparams):
    return Centaur(hparams)
