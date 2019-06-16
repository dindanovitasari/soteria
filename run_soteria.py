# -*- coding: utf-8 -*-
"""

@author: dinda
python 3.6.8
"""
import argparse
from soteria import Soteria

parser = argparse.ArgumentParser(description='Run Soteria Engine')
parser.add_argument('feature_dir', type=str, help='Input directory for features')
parser.add_argument('label_dir', type=str, help='Input directory for labels')
args = parser.parse_args()

Soteria().run_engine(args.feature_dir, args.label_dir)