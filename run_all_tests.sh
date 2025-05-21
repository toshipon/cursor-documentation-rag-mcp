#!/bin/bash
# 全テスト（単体テストと統合テスト）を順番に実行するスクリプト

# スクリプト終了時にメッセージを表示する関数
function finish {
  echo "======================================"
  if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed successfully!"
  else
    echo "❌ Some tests failed. Please check the logs."
  fi
  echo "======================================"
}

# 初期設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

OVERALL_EXIT_CODE=0
trap finish EXIT

# バナー表示
echo "======================================"
echo "Memory Bank Project - Test Suite"
echo "$(date)"
echo "======================================"

# 1. 単体テストの実行
echo "[1/2] Running unit tests..."
python run_unit_tests.py -v
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed"
    OVERALL_EXIT_CODE=1
else
    echo "✅ Unit tests completed successfully"
fi

# 2. インテグレーションテストの実行
echo
echo "[2/2] Running integration tests..."
./run_tests.sh
if [ $? -ne 0 ]; then
    echo "❌ Integration tests failed"
    OVERALL_EXIT_CODE=1
else
    echo "✅ Integration tests completed successfully"
fi

exit $OVERALL_EXIT_CODE
