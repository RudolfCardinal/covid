#!/bin/bash
set -e

function rm_file_from_history {
    local filename=$1
    echo "Removing ${filename}"

    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch ${filename}" \
        --prune-empty --tag-name-filter cat -- --all
}

THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${THIS_SCRIPT_DIR}"

rm_file_from_history spread_modelling/saved_results/saved_output_2020_04_21_1428.zip
rm_file_from_history spread_modelling/saved_results/saved_output_2020_04_26.zip
rm_file_from_history spread_modelling/saved_results/saved_results_2020_04_20.zip
rm_file_from_history spread_modelling/saved_results/saved_results_2020_04_26.zip
rm_file_from_history spread_modelling/saved_results/saved_results_2020_04_08/exp1_daily.csv

# See:
# https://help.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository
#
# Now do:
#
# git push github --force --all
