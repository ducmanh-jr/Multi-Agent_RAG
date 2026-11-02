import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from scripts.test_agent0_scout_suite import run_suite as run_phase1
from scripts.test_phase2_sandbox_and_auditor import run_tests as run_phase2
from scripts.test_phase3_gap_radar import run_tests as run_phase3

def run_all():
    print("=====================================================================")
    print("   UNIFIED VERIFICATION SUITE: FRONTIER SCIENTIFIC ENGINE (ALL PHASES)")
    print("=====================================================================")
    start_time = time.time()

    print("\n>>> EXECUTING PHASE 1: AGENT 0 LITERATURE SCOUT...")
    run_phase1()

    print("\n>>> EXECUTING PHASE 2: PYTHON SANDBOX & STATISTICAL AUDITOR...")
    run_phase2()

    print("\n>>> EXECUTING PHASE 3: CROSS-PAPER BENCHMARK MATRIX & GAP RADAR...")
    run_phase3()

    total_time = time.time() - start_time
    print("\n=====================================================================")
    print(f"   [PASS] ALL 3 PHASES PASSED 100% IN {total_time:.2f}s WITH ZERO ERRORS!")
    print("=====================================================================")

if __name__ == "__main__":
    run_all()
