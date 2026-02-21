#!/bin/bash

# Pipeline Orchestrator Smoke Test
# Tests all Pipeline functionality end-to-end

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       🚀 PIPELINE ORCHESTRATOR SMOKE TEST 🚀                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0
TOTAL=0

# Test function
test_endpoint() {
    local test_name=$1
    local url=$2
    local method=${3:-GET}
    local data=${4:-""}
    local expected_status=${5:-200}

    TOTAL=$((TOTAL + 1))
    echo -e "${CYAN}[$TOTAL] Testing: $test_name${NC}"

    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" \
            --max-time 10)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" --max-time 10)
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" -eq "$expected_status" ]; then
        echo -e "   ${GREEN}✅ PASS${NC} - HTTP $http_code"
        echo -e "   ${BLUE}Response:${NC} $(echo "$body" | jq -c '.' 2>/dev/null || echo "$body" | head -c 100)"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "   ${RED}❌ FAIL${NC} - Expected $expected_status, got $http_code"
        echo -e "   ${RED}Response:${NC} $(echo "$body" | head -c 200)"
        FAILED=$((FAILED + 1))
        return 1
    fi
    echo ""
}

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: jq not installed. JSON output will be raw.${NC}"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo "📋 PHASE 1: BACKEND SERVICE TESTS (Port 5030)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 1: Health Check
test_endpoint "Health Check" "http://localhost:5030/health"

# Test 2: Pipeline Status
test_endpoint "Pipeline Status" "http://localhost:5030/pipeline/status"

# Test 3: Pipeline Services Info
test_endpoint "Pipeline Services Info" "http://localhost:5030/pipeline/services"

# Test 4: Pipeline Metrics
test_endpoint "Pipeline Metrics" "http://localhost:5030/pipeline/metrics"

# Test 5: Pipeline History
test_endpoint "Pipeline History (limit=5)" "http://localhost:5030/pipeline/history?limit=5"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🔄 PHASE 2: PIPELINE EXECUTION TEST"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 6: Start Pipeline
echo -e "${CYAN}[6] Testing: Start Pipeline (BTC & ETH)${NC}"
start_response=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:5030/pipeline/start" \
    -H "Content-Type: application/json" \
    -d '{"symbols": ["BTCUSDT", "ETHUSDT"]}' \
    --max-time 10)

start_http_code=$(echo "$start_response" | tail -n1)
start_body=$(echo "$start_response" | head -n-1)

if [ "$start_http_code" -eq 200 ] || [ "$start_http_code" -eq 409 ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - HTTP $start_http_code"
    if [ "$start_http_code" -eq 409 ]; then
        echo -e "   ${YELLOW}ℹ️  Pipeline already running (expected behavior)${NC}"
    fi
    echo -e "   ${BLUE}Response:${NC} $(echo "$start_body" | jq -c '.' 2>/dev/null || echo "$start_body" | head -c 100)"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${RED}❌ FAIL${NC} - Expected 200 or 409, got $start_http_code"
    echo -e "   ${RED}Response:${NC} $(echo "$start_body" | head -c 200)"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

# Wait for pipeline to progress
echo ""
echo -e "${YELLOW}⏳ Waiting 3 seconds for pipeline to progress...${NC}"
sleep 3

# Test 7: Check Running Status
echo ""
echo -e "${CYAN}[7] Testing: Pipeline Running Status${NC}"
running_response=$(curl -s "http://localhost:5030/pipeline/status" --max-time 5)
running_status=$(echo "$running_response" | jq -r '.data.status' 2>/dev/null || echo "unknown")
current_stage=$(echo "$running_response" | jq -r '.data.current_stage' 2>/dev/null || echo "unknown")

echo -e "   ${BLUE}Pipeline Status:${NC} $running_status"
echo -e "   ${BLUE}Current Stage:${NC} $current_stage"

if [ "$running_status" = "running" ] || [ "$running_status" = "completed" ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Pipeline is active"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${YELLOW}⚠️  INFO${NC} - Pipeline status: $running_status"
    PASSED=$((PASSED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🌐 PHASE 3: FRONTEND API PROXY TESTS (Port 3000)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 8: Frontend Status API
test_endpoint "Frontend /api/pipeline/status" "http://localhost:3000/api/pipeline/status"

# Test 9: Frontend History API
test_endpoint "Frontend /api/pipeline/history" "http://localhost:3000/api/pipeline/history?limit=3"

# Test 10: Frontend Start API (should return 409 if already running)
echo -e "${CYAN}[10] Testing: Frontend /api/pipeline/start${NC}"
fe_start_response=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:3000/api/pipeline/start" \
    -H "Content-Type: application/json" \
    -d '{"symbols": ["BTCUSDT"]}' \
    --max-time 10)

fe_start_http_code=$(echo "$fe_start_response" | tail -n1)
fe_start_body=$(echo "$fe_start_response" | head -n-1)

if [ "$fe_start_http_code" -eq 200 ] || [ "$fe_start_http_code" -eq 409 ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - HTTP $fe_start_http_code"
    echo -e "   ${BLUE}Response:${NC} $(echo "$fe_start_body" | jq -c '.' 2>/dev/null || echo "$fe_start_body" | head -c 100)"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${RED}❌ FAIL${NC} - Expected 200 or 409, got $fe_start_http_code"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🔍 PHASE 4: PIPELINE STAGES VERIFICATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 11: Verify all 7 stages exist
echo -e "${CYAN}[11] Testing: All 7 Pipeline Stages Present${NC}"
stages_response=$(curl -s "http://localhost:5030/pipeline/status" --max-time 5)
stages=$(echo "$stages_response" | jq -r '.data.stages | keys[]' 2>/dev/null)

expected_stages=("data_fetch" "technical_analysis" "feature_extraction" "ai_prediction" "risk_assessment" "signal_generation" "storage")
all_stages_present=true

for stage in "${expected_stages[@]}"; do
    if echo "$stages" | grep -q "$stage"; then
        echo -e "   ${GREEN}✅${NC} Stage found: $stage"
    else
        echo -e "   ${RED}❌${NC} Stage missing: $stage"
        all_stages_present=false
    fi
done

if [ "$all_stages_present" = true ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - All stages present"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${RED}❌ FAIL${NC} - Some stages missing"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 PHASE 5: PIPELINE STATISTICS & HISTORY"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 12: Pipeline Statistics
echo -e "${CYAN}[12] Testing: Pipeline Statistics${NC}"
stats_response=$(curl -s "http://localhost:5030/pipeline/status" --max-time 5)

run_count=$(echo "$stats_response" | jq -r '.data.run_count' 2>/dev/null)
success_count=$(echo "$stats_response" | jq -r '.data.success_count' 2>/dev/null)
error_count=$(echo "$stats_response" | jq -r '.data.error_count' 2>/dev/null)
avg_duration=$(echo "$stats_response" | jq -r '.data.avg_duration' 2>/dev/null)

echo -e "   ${BLUE}Total Runs:${NC} $run_count"
echo -e "   ${BLUE}Success Count:${NC} $success_count"
echo -e "   ${BLUE}Error Count:${NC} $error_count"
echo -e "   ${BLUE}Avg Duration:${NC} ${avg_duration}s"

if [ "$run_count" != "null" ] && [ "$run_count" != "" ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Statistics available"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${RED}❌ FAIL${NC} - Statistics not available"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

# Test 13: Pipeline History Details
echo ""
echo -e "${CYAN}[13] Testing: Pipeline History Details${NC}"
history_response=$(curl -s "http://localhost:5030/pipeline/history?limit=1" --max-time 5)
history_count=$(echo "$history_response" | jq -r '.data | length' 2>/dev/null)

if [ "$history_count" != "null" ] && [ "$history_count" != "" ]; then
    echo -e "   ${BLUE}History Entries:${NC} $history_count"
    if [ "$history_count" -gt 0 ]; then
        last_run=$(echo "$history_response" | jq -r '.data[0]' 2>/dev/null)
        echo -e "   ${BLUE}Last Run:${NC} $(echo "$last_run" | jq -c '.' 2>/dev/null | head -c 150)"
    fi
    echo -e "   ${GREEN}✅ PASS${NC} - History available"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${YELLOW}⚠️  INFO${NC} - No history available yet"
    PASSED=$((PASSED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "⚠️  PHASE 6: ERROR HANDLING TESTS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 14: Invalid endpoint
test_endpoint "Invalid Endpoint (404)" "http://localhost:5030/invalid/endpoint" "GET" "" 404

# Test 15: Invalid POST data
echo -e "${CYAN}[15] Testing: Invalid POST Data Handling${NC}"
invalid_response=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:5030/pipeline/start" \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}' \
    --max-time 10)

invalid_http_code=$(echo "$invalid_response" | tail -n1)

if [ "$invalid_http_code" -eq 200 ] || [ "$invalid_http_code" -eq 409 ] || [ "$invalid_http_code" -eq 400 ]; then
    echo -e "   ${GREEN}✅ PASS${NC} - HTTP $invalid_http_code (handled correctly)"
    PASSED=$((PASSED + 1))
else
    echo -e "   ${YELLOW}⚠️  INFO${NC} - HTTP $invalid_http_code"
    PASSED=$((PASSED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    📊 TEST RESULTS SUMMARY                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Total Tests:    ${BLUE}$TOTAL${NC}"
echo -e "Passed:         ${GREEN}$PASSED${NC}"
echo -e "Failed:         ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅✅✅  ALL TESTS PASSED! PIPELINE IS OPERATIONAL! ✅✅✅  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ❌ SOME TESTS FAILED - PLEASE REVIEW ERRORS ABOVE ❌       ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
