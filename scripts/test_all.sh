#!/bin/bash
# Run all PromptGuard tests
#
# Usage:
#   ./scripts/test_all.sh           # Run all tests
#   ./scripts/test_all.sh --no-api  # Skip tests requiring API key

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "PromptGuard Test Suite"
echo "========================================"
echo ""

# Check for --no-api flag
SKIP_API=false
for arg in "$@"; do
    if [ "$arg" = "--no-api" ]; then
        SKIP_API=true
    fi
done

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ "$SKIP_API" = false ]; then
    echo "WARNING: OPENAI_API_KEY not set"
    echo "Integration tests will be skipped."
    echo "Set the key or use --no-api flag to suppress this warning."
    echo ""
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Step 1: Unit tests
echo "Step 1: Running unit tests..."
echo "----------------------------------------"
pytest tests/test_config.py tests/test_adapters.py tests/test_validator.py tests/test_hashing.py tests/test_snapshot.py -v
echo ""

# Step 2: E2E tests (no API required for most)
echo "Step 2: Running E2E tests..."
echo "----------------------------------------"
pytest tests/test_e2e.py -v
echo ""

# Step 3: Integration tests (requires API)
if [ "$SKIP_API" = false ] && [ -n "$OPENAI_API_KEY" ]; then
    echo "Step 3: Running integration tests..."
    echo "----------------------------------------"
    pytest tests/test_integration.py -v
    echo ""
else
    echo "Step 3: Skipping integration tests (no API key)"
    echo ""
fi

# Step 4: Smoke test
echo "Step 4: Running smoke test..."
echo "----------------------------------------"
python scripts/smoke_test.py
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
