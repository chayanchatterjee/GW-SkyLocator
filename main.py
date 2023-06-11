# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.GW_SkyLocator import GW_SkyLocator


def run():
    """Builds model, loads data, trains and evaluates"""
    model = GW_SkyLocator(CFG)
    model.load_data()
#    model.load_test_data()
    model.construct_model()
#    model.train()
    model.obtain_samples()


if __name__ == '__main__':
    run()
