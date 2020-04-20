#!/usr/bin/env python

"""
slurm_submit_spreadmodel.py

===============================================================================

    Copyright (C) 2020-2020 Rudolf Cardinal (rudolf@pobox.com).

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software. If not, see <http://www.gnu.org/licenses/>.

===============================================================================

Note some oddities:

- I was watching a process with ``tail -f slurm-JOBID.out`` via SSH, and it
  died somehow and appeared to leave the process running (according to SLURM)
  but not producing anything. Something to avoid? Intermittent ``tail`` seems
  fine. Coincidence? Got stuck for some other reason?

  Ah. Hit memory limit (48 CPUs, 96 Gb RAM) after 141 iterations, on another
  occasion.

  Shift to https://pypi.org/project/bounded-pool-executor/ instead.

  Nope; shift to ``parallelize_processes_efficiently()`` function.

- The CSV output files get created within a few seconds, but are zero-length
  for a long time... probably fixed with
  ``parallelize_processes_efficiently()``.

- Our longest experiment, Experiment 2, takes under 48h on a 3 GHz processor
  using 8 effective cores. So we anticipate about 8 hours with 48 cores,
  though perhaps the HPHI ones aren't quite as fast. We could choose ``day.q``
  with its 1-day limit, or ``short.q`` with its 8-hour limit.

"""

import argparse
from datetime import timedelta
import logging
import os
import sys

from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger
from cardinal_pythonlib.slurm import launch_cambridge_hphi

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable  # we need the correct venv version
SCRIPT = os.path.join(THIS_DIR, "spread_modelling.py")


def launch(job: str) -> None:
    jobname = f"spread_{job}"
    cmd = f"{PYTHON} {SCRIPT} {job}"
    launch_cambridge_hphi(
        jobname=jobname,
        cmd=cmd,
        memory_mb=48000,
        qos="day.q",
        # qos="short.q",
        email="rnc1001@cam.ac.uk",
        duration=timedelta(days=1),
        # duration=timedelta(hours=8),
        cpus_per_task=24,  # 48 works but competition++; 24 on instantly
        partition="wbic-cs"
    )


def main() -> None:
    main_only_quicksetup_rootlogger(level=logging.DEBUG)

    # Define jobs
    possible_jobs = ["experiment_1", "experiment_1b", "experiment_2"]

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Launch spread-modelling job on SLURM cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "jobs", type=str, nargs="+", choices=possible_jobs + ["all"],
        help="Job code")
    args = parser.parse_args()

    # Launch
    jobs = possible_jobs if "all" in args.jobs else args.jobs
    for job in jobs:
        launch(job)


if __name__ == '__main__':
    main()
