#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
単体テスト実行スクリプト

使用方法:
  python run_unit_tests.py
"""

import os
import sys
import unittest
import argparse
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests(test_pattern=None, verbose=False):
    """
    単体テストを実行
    
    Args:
        test_pattern: 特定のテストだけを実行する場合のパターン
        verbose: 詳細出力するかどうか
    """
    # テストディレクトリ
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    
    # テストを探索して実行
    if test_pattern:
        logger.info(f"Running tests matching pattern: {test_pattern}")
        test_suite = unittest.defaultTestLoader.discover(test_dir, pattern=test_pattern)
    else:
        logger.info("Running all unit tests")
        test_suite = unittest.defaultTestLoader.discover(test_dir, pattern="unit_test_*.py")
    
    # テスト実行
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    # 結果集計
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    # テスト失敗時は終了コード1で終了
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("-p", "--pattern", help="Test file pattern to run (e.g. unit_test_vector_store.py)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    exit_code = run_tests(args.pattern, args.verbose)
    sys.exit(exit_code)
