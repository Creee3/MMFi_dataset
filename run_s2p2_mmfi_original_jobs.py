"""
Run S2P2 jobs with the MMFi original held-out validation protocol.

This wrapper intentionally disables the extra held-out val/test split and saves
results into a separate directory so earlier HPE-Li-style or strict-subject
runs are not overwritten.
"""

import os
import sys

from run_s2p2_selected_jobs import main as selected_main
from run_s2p2_wifi_cmc_mix20 import RESULTS_ROOT


DEFAULT_JOBS = "1,4,5,7,8"
DEFAULT_RESULTS_DIR = os.path.join(
    RESULTS_ROOT, "S2P2_wifi_cmc_mix20_mmfi_original")


def has_option(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


if __name__ == "__main__":
    os.environ["MMFI_PROTOCOL"] = "protocol2"
    os.environ["HELDOUT_SPLIT_ENABLED"] = "0"
    if not has_option("--results_root"):
        sys.argv.extend(["--results_root", DEFAULT_RESULTS_DIR])
    if not has_option("--jobs"):
        sys.argv.extend(["--jobs", DEFAULT_JOBS])
    selected_main()
