#!/bin/sh
set -x
eval "$(conda shell.bash hook)"
conda activate sembre
for QUERY_N in 1 3 5; do
	for LAYER in {0..11}; do
		for QUERY_CATEGORY in "all" "nota" "non-nota"; do
			for POS in "v" "n" "all"; do
				python ontonotes_main.py \
					--metric cosine \
					--query-n $QUERY_N \
					--bert-layers $LAYER \
					--query-category $QUERY_CATEGORY \
					--pos $POS
			done
		done
	done
done
