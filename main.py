# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.GW_SkyNet import GW_SkyNet


def run():
    """Builds model, loads data, trains and evaluates"""
    model = GW_SkyNet(CFG)
    model.load_data()
#    model.load_test_data()
    model.construct_model()
#    model.train()
    model.obtain_samples()


if __name__ == '__main__':
    run()
