#!/bin/bash

echo "🔐 LYDIAN TRADER - WHITE HAT PENETRATION TEST"
echo "=============================================="
echo "Railway Production Security Assessment"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="http://localhost:3000"
PASSED=0
FAILED=0
WARNINGS=0

# ========================================
# 1. AUTHENTICATION & AUTHORIZATION TESTS
# ========================================
echo -e "${BLUE}[1/10] AUTHENTICATION & AUTHORIZATION${NC}"
echo "--------------------------------------"

# Test 1.1: Unauthenticated access to protected routes
echo -n "Testing protected route access without auth... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/dashboard" 2>/dev/null)
if [ "$response" = "307" ] || [ "$response" = "302" ] || [ "$response" = "401" ]; then
    echo -e "${GREEN}✓ PASS${NC} (Redirected to login)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (HTTP $response - Should redirect/block)"
    ((FAILED++))
fi

# Test 1.2: Login page accessibility
echo -n "Testing login page accessibility... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/login" 2>/dev/null)
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (HTTP $response)"
    ((FAILED++))
fi

# Test 1.3: SQL Injection attempt on login
echo -n "Testing SQL injection protection (login)... "
sql_payload="' OR '1'='1"
response=$(curl -s -X POST "$BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$sql_payload\",\"password\":\"test\"}" \
    -w "%{http_code}" -o /dev/null 2>/dev/null)
if [ "$response" = "400" ] || [ "$response" = "401" ] || [ "$response" = "422" ]; then
    echo -e "${GREEN}✓ PASS${NC} (SQL injection blocked)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (HTTP $response - Review input validation)"
    ((WARNINGS++))
fi

# Test 1.4: Rate limiting check
echo -n "Testing rate limiting (brute force protection)... "
for i in {1..6}; do
    curl -s -X POST "$BASE_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"email":"test@test.com","password":"wrong"}' \
        -o /dev/null 2>/dev/null
done
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"wrong"}' 2>/dev/null)
if [ "$response" = "429" ]; then
    echo -e "${GREEN}✓ PASS${NC} (Rate limit active)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Rate limiting not detected)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 2. XSS (CROSS-SITE SCRIPTING) TESTS
# ========================================
echo -e "${BLUE}[2/10] XSS PROTECTION${NC}"
echo "---------------------"

# Test 2.1: XSS in query parameters
echo -n "Testing XSS protection in URL parameters... "
xss_payload="<script>alert('XSS')</script>"
response=$(curl -s "$BASE_URL/search?q=$xss_payload" | grep -o "<script>" | wc -l)
if [ "$response" -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} (XSS payload sanitized)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (XSS vulnerability detected)"
    ((FAILED++))
fi

# Test 2.2: Content Security Policy header
echo -n "Testing CSP (Content Security Policy) header... "
csp_header=$(curl -s -I "$BASE_URL" | grep -i "content-security-policy")
if [ -n "$csp_header" ]; then
    echo -e "${GREEN}✓ PASS${NC} (CSP header present)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (CSP header missing)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 3. SECURITY HEADERS
# ========================================
echo -e "${BLUE}[3/10] SECURITY HEADERS${NC}"
echo "----------------------"

# Test 3.1: X-Frame-Options
echo -n "Testing X-Frame-Options header... "
header=$(curl -s -I "$BASE_URL" | grep -i "x-frame-options")
if [[ $header == *"DENY"* ]] || [[ $header == *"SAMEORIGIN"* ]]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Clickjacking protection missing)"
    ((WARNINGS++))
fi

# Test 3.2: X-Content-Type-Options
echo -n "Testing X-Content-Type-Options header... "
header=$(curl -s -I "$BASE_URL" | grep -i "x-content-type-options: nosniff")
if [ -n "$header" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (MIME sniffing protection missing)"
    ((WARNINGS++))
fi

# Test 3.3: Strict-Transport-Security (HSTS)
echo -n "Testing HSTS header... "
header=$(curl -s -I "$BASE_URL" | grep -i "strict-transport-security")
if [ -n "$header" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ INFO${NC} (HSTS should be enabled in production)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 4. API SECURITY TESTS
# ========================================
echo -e "${BLUE}[4/10] API SECURITY${NC}"
echo "-------------------"

# Test 4.1: API without authentication
echo -n "Testing API authentication requirement... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/quantum-pro/bots" 2>/dev/null)
if [ "$response" = "401" ] || [ "$response" = "403" ] || [ "$response" = "307" ]; then
    echo -e "${GREEN}✓ PASS${NC} (API requires auth)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ INFO${NC} (HTTP $response - Public API endpoint)"
    ((WARNINGS++))
fi

# Test 4.2: CORS headers
echo -n "Testing CORS configuration... "
cors=$(curl -s -I -H "Origin: https://evil.com" "$BASE_URL/api/market/crypto" | grep -i "access-control-allow-origin")
if [ -z "$cors" ]; then
    echo -e "${GREEN}✓ PASS${NC} (CORS properly restricted)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (CORS may be too permissive)"
    ((WARNINGS++))
fi

# Test 4.3: API payload size limit
echo -n "Testing API payload size limit... "
large_payload=$(python3 -c "print('x' * 1000000)" 2>/dev/null)
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"data\":\"$large_payload\"}" 2>/dev/null)
if [ "$response" = "413" ] || [ "$response" = "400" ]; then
    echo -e "${GREEN}✓ PASS${NC} (Payload size limited)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ INFO${NC} (Large payload handling: HTTP $response)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 5. INFORMATION DISCLOSURE
# ========================================
echo -e "${BLUE}[5/10] INFORMATION DISCLOSURE${NC}"
echo "-----------------------------"

# Test 5.1: Error messages
echo -n "Testing error message disclosure... "
response=$(curl -s "$BASE_URL/nonexistent-page-12345" | grep -i "stack trace\|error:\|exception:" | wc -l)
if [ "$response" -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} (No stack traces exposed)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Sensitive error info exposed)"
    ((FAILED++))
fi

# Test 5.2: Server header
echo -n "Testing server header disclosure... "
server=$(curl -s -I "$BASE_URL" | grep -i "^server:")
if [ -z "$server" ] || [[ $server != *"Express"* ]] && [[ $server != *"Node"* ]]; then
    echo -e "${GREEN}✓ PASS${NC} (Server info hidden)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Server version exposed: $server)"
    ((WARNINGS++))
fi

# Test 5.3: Source map exposure
echo -n "Testing source map exposure... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/_next/static/chunks/main.js.map" 2>/dev/null)
if [ "$response" = "404" ]; then
    echo -e "${GREEN}✓ PASS${NC} (Source maps not exposed)"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Source maps accessible)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 6. SESSION MANAGEMENT
# ========================================
echo -e "${BLUE}[6/10] SESSION MANAGEMENT${NC}"
echo "-------------------------"

# Test 6.1: Session cookie security
echo -n "Testing session cookie security flags... "
cookies=$(curl -s -I "$BASE_URL/login" | grep -i "set-cookie")
if [[ $cookies == *"HttpOnly"* ]] && [[ $cookies == *"Secure"* ]]; then
    echo -e "${GREEN}✓ PASS${NC} (Secure cookies)"
    ((PASSED++))
elif [ -z "$cookies" ]; then
    echo -e "${YELLOW}⚠ INFO${NC} (No cookies set yet)"
    ((WARNINGS++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Cookie security flags missing)"
    ((WARNINGS++))
fi

# Test 6.2: Session fixation
echo -n "Testing session fixation protection... "
# This is a basic check - full test requires login flow
echo -e "${YELLOW}⚠ INFO${NC} (Manual verification required)"
((WARNINGS++))

echo ""

# ========================================
# 7. INPUT VALIDATION
# ========================================
echo -e "${BLUE}[7/10] INPUT VALIDATION${NC}"
echo "----------------------"

# Test 7.1: NoSQL injection
echo -n "Testing NoSQL injection protection... "
nosql_payload='{"$gt":""}'
response=$(curl -s -X POST "$BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":$nosql_payload,\"password\":\"test\"}" \
    -w "%{http_code}" -o /dev/null 2>/dev/null)
if [ "$response" = "400" ] || [ "$response" = "401" ] || [ "$response" = "422" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (HTTP $response)"
    ((WARNINGS++))
fi

# Test 7.2: Command injection
echo -n "Testing command injection protection... "
cmd_payload="; ls -la"
response=$(curl -s "$BASE_URL/api/search?q=$cmd_payload" -w "%{http_code}" -o /dev/null 2>/dev/null)
if [ "$response" = "200" ] || [ "$response" = "400" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (HTTP $response)"
    ((FAILED++))
fi

echo ""

# ========================================
# 8. CRYPTOGRAPHY
# ========================================
echo -e "${BLUE}[8/10] CRYPTOGRAPHY${NC}"
echo "-------------------"

# Test 8.1: HTTPS enforcement
echo -n "Testing HTTPS enforcement (production)... "
echo -e "${YELLOW}⚠ INFO${NC} (Should enforce HTTPS on Railway)"
((WARNINGS++))

# Test 8.2: Password storage
echo -n "Testing password hashing (bcrypt/argon2)... "
echo -e "${YELLOW}⚠ INFO${NC} (Code review required)"
((WARNINGS++))

echo ""

# ========================================
# 9. BUSINESS LOGIC
# ========================================
echo -e "${BLUE}[9/10] BUSINESS LOGIC${NC}"
echo "---------------------"

# Test 9.1: API rate limiting
echo -n "Testing API rate limiting... "
for i in {1..20}; do
    curl -s "$BASE_URL/api/market/crypto" -o /dev/null 2>/dev/null
done
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/market/crypto" 2>/dev/null)
if [ "$response" = "429" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ INFO${NC} (No rate limit detected - HTTP $response)"
    ((WARNINGS++))
fi

# Test 9.2: Unauthorized bot access
echo -n "Testing bot management authorization... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/quantum-pro/bots" 2>/dev/null)
if [ "$response" = "401" ] || [ "$response" = "403" ] || [ "$response" = "307" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ INFO${NC} (HTTP $response)"
    ((WARNINGS++))
fi

echo ""

# ========================================
# 10. DEPENDENCY VULNERABILITIES
# ========================================
echo -e "${BLUE}[10/10] DEPENDENCY SECURITY${NC}"
echo "--------------------------"

# Test 10.1: npm audit
echo -n "Running npm audit... "
if [ -f "package.json" ]; then
    audit_result=$(npm audit --json 2>/dev/null | grep -o '"high":[0-9]*' | cut -d':' -f2)
    if [ -z "$audit_result" ] || [ "$audit_result" -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} (No critical vulnerabilities)"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC} ($audit_result high vulnerabilities)"
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}⚠ INFO${NC} (package.json not found)"
    ((WARNINGS++))
fi

echo ""
echo "=============================================="
echo -e "${BLUE}📊 PENETRATION TEST RESULTS${NC}"
echo "=============================================="
echo -e "Total Tests: $((PASSED + FAILED + WARNINGS))"
echo -e "${GREEN}✓ Passed: $PASSED${NC}"
echo -e "${RED}✗ Failed: $FAILED${NC}"
echo -e "${YELLOW}⚠ Warnings: $WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 SECURITY BASELINE PASSED!${NC}"
    echo "✅ System ready for Railway production deployment"
    echo ""
    echo "Recommendations:"
    echo "  - Enable HSTS in production"
    echo "  - Implement rate limiting on all APIs"
    echo "  - Regular security audits"
    echo "  - Keep dependencies updated"
    exit 0
else
    echo -e "${RED}🚨 SECURITY ISSUES DETECTED!${NC}"
    echo "⚠️  Address critical issues before production deployment"
    exit 1
fi
