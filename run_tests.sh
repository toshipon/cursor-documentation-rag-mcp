#!/bin/bash
# テスト実行スクリプト

# 色付きの出力
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ヘルパー関数
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# レポートディレクトリの作成
mkdir -p reports

# 引数の処理
SKIP_INTEGRATION=false
SKIP_LOAD=false
SKIP_RESOURCE=false
SKIP_GPU=false
NO_DOCKER=false

for arg in "$@"; do
    case $arg in
        --skip-integration)
            SKIP_INTEGRATION=true
            ;;
        --skip-load)
            SKIP_LOAD=true
            ;;
        --skip-resource)
            SKIP_RESOURCE=true
            ;;
        --skip-gpu)
            SKIP_GPU=true
            ;;
        --no-docker)
            NO_DOCKER=true
            ;;
    esac
done

# 依存パッケージのインストールを確認
print_header "依存パッケージの確認"
python -m pip install -q pytest tqdm matplotlib locust

# 1. インテグレーションテスト
if [ "$SKIP_INTEGRATION" = false ]; then
    print_header "インテグレーションテストの実行"
    
    if [ "$NO_DOCKER" = true ]; then
        print_warning "Dockerを使用せずにテストを実行します（モック実装を使用）"
        export NO_DOCKER_TEST=1
    fi
    
    python -m pytest tests/integration_test.py -v
    TEST_EXIT_CODE=$?
    
    # 環境変数のリセット
    unset NO_DOCKER_TEST
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "インテグレーションテスト完了"
    else
        print_error "インテグレーションテストに失敗しました"
    fi
else
    print_warning "インテグレーションテストをスキップします"
fi

# 2. リソース検証テスト
if [ "$SKIP_RESOURCE" = false ]; then
    if [ "$NO_DOCKER" = true ]; then
        print_warning "リソース検証テストはDockerが必要なためスキップします"
    else
        print_header "リソース検証テストの実行"
        echo "MCPサーバーのリソース使用状況を5分間監視します..."
        python tests/resource_validation.py --duration 5 --containers mcp-server
        if [ $? -eq 0 ]; then
            print_success "リソース検証テスト完了"
        else
            print_error "リソース検証テストに失敗しました"
        fi
    fi
else
    print_warning "リソース検証テストをスキップします"
fi

# 3. 負荷テスト
if [ "$SKIP_LOAD" = false ]; then
    if [ "$NO_DOCKER" = true ]; then
        print_warning "負荷テストはDockerが必要なためスキップします"
    else
        print_header "負荷テストの実行"
        echo "MCPサーバーに対する負荷テストを開始します..."
        python tests/load_test.py --users 10 --time 30s
        if [ $? -eq 0 ]; then
            print_success "負荷テスト完了"
        else
            print_error "負荷テストに失敗しました"
        fi
    fi
else
    print_warning "負荷テストをスキップします"
fi

# 4. GPU検証
if [ "$SKIP_GPU" = false ]; then
    if [ "$NO_DOCKER" = true ]; then
        print_warning "GPU検証はDockerが必要なためスキップします"
    else
        print_header "GPU検証の実行"
        echo "CPUとGPUのパフォーマンス比較テストを実行します..."
        python tests/gpu_validation.py --samples 50
        if [ $? -eq 0 ]; then
            print_success "GPU検証完了"
        else
            print_error "GPU検証に失敗しました"
        fi
    fi
else
    print_warning "GPU検証をスキップします"
fi

print_header "テスト概要"
echo "テストレポートは reports/ ディレクトリに保存されています"
echo "全テスト完了しました"