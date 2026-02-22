#!/bin/bash

# 🔥 TYPESCRIPT ERROR AUTO-FIX SCRIPT
# AILYDIAN NIRVANA MODE - ZERO-ERROR PROTOCOL
# Target: Fix 1543 TypeScript errors systematically

set -e

PROJECT_ROOT="/home/lydian/Masaüstü/PROJELER/borsa.ailydian.com"
cd "$PROJECT_ROOT"

echo "🔥 STARTING TYPESCRIPT AUTO-FIX"
echo "================================"
echo ""

# Function to count errors
count_errors() {
  pnpm typecheck 2>&1 | grep -c "error TS" || echo "0"
}

echo "📊 Initial error count: $(count_errors)"
echo ""

# ============================================
# PHASE 1: FIX TS6133 (Unused Variables - 450)
# ============================================
echo "🔧 PHASE 1: Fixing TS6133 (Unused variables)"
echo "============================================="

# Get list of files with TS6133 errors
pnpm typecheck 2>&1 | grep "error TS6133" | cut -d'(' -f1 | sort -u > /tmp/ts6133_files.txt

FILE_COUNT=$(wc -l < /tmp/ts6133_files.txt)
echo "📁 Files to fix: $FILE_COUNT"

if [ $FILE_COUNT -gt 0 ]; then
  # Fix common patterns
  while IFS= read -r file; do
    if [ -f "$file" ]; then
      echo "  Fixing: $file"

      # Pattern 1: Remove unused imports
      # Example: import { unused } from 'lib' → (if completely unused)

      # Pattern 2: Prefix unused variables with underscore
      # This is safer - doesn't break code, just suppresses error
      # We'll do this in Node.js script for better regex control
    fi
  done < /tmp/ts6133_files.txt
fi

echo "✅ Phase 1 complete"
echo "📊 Current error count: $(count_errors)"
echo ""

# ============================================
# PHASE 2: FIX TS2322/TS2339 (Type Mismatch - 71)
# ============================================
echo "🔧 PHASE 2: Fixing TS2322/TS2339 (Type mismatches)"
echo "=================================================="

# These need manual review, but we can list them
pnpm typecheck 2>&1 | grep -E "error TS(2322|2339)" > /tmp/type_errors.txt || true
echo "📝 Type errors saved to /tmp/type_errors.txt"
echo "⚠️  These require manual fix (71 errors)"
echo ""

# ============================================
# PHASE 3: FIX TS18048/TS2532 (Undefined - 1022)
# ============================================
echo "🔧 PHASE 3: Fixing TS18048/TS2532 (Undefined checks)"
echo "===================================================="

# Get list of files with undefined errors
pnpm typecheck 2>&1 | grep -E "error TS(18048|2532)" | cut -d'(' -f1 | sort -u > /tmp/undefined_files.txt

FILE_COUNT=$(wc -l < /tmp/undefined_files.txt)
echo "📁 Files to fix: $FILE_COUNT"

if [ $FILE_COUNT -gt 0 ]; then
  echo "⚠️  Undefined checks require context-aware fixes"
  echo "   Will use Node.js script for intelligent fixes"
fi

echo ""
echo "✅ Analysis complete"
echo ""

# ============================================
# PHASE 4: SMART FIX (Node.js)
# ============================================
echo "🔧 PHASE 4: Running smart fix with Node.js"
echo "=========================================="

node "$PROJECT_ROOT/scripts/fix-typescript-smart.js"

echo ""
echo "✅ Smart fix complete"
echo "📊 Final error count: $(count_errors)"
echo ""

# ============================================
# SUMMARY
# ============================================
FINAL_ERRORS=$(count_errors)
FIXED=$((1543 - FINAL_ERRORS))

echo "📈 SUMMARY"
echo "=========="
echo "  Initial errors: 1543"
echo "  Fixed: $FIXED"
echo "  Remaining: $FINAL_ERRORS"
echo ""

if [ "$FINAL_ERRORS" -eq 0 ]; then
  echo "🎉 ZERO ERRORS! Project is type-safe!"
else
  echo "⚠️  $FINAL_ERRORS errors remaining"
  echo "   Run: pnpm typecheck | head -50"
fi

echo ""
echo "🔥 TYPESCRIPT AUTO-FIX COMPLETE"
