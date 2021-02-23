#!/bin/sh
set -x
eval "$(conda shell.bash hook)"
conda activate sembre
for QUERY_CATEGORY in "all" "nota" "non-nota"; do
	for POS in "v" "n" "all"; do
		python ontonotes_main.py \
			--metric baseline \
			--query-category $QUERY_CATEGORY \
			--pos $POS
	done
done
