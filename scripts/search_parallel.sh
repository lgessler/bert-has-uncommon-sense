#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sembre
FILENAME="/tmp/bssp_commands_parallel.txt"
rm -rf $FILENAME
for QUERY_N in 1 3 5; do
	for LAYER in {0..11}; do
		for QUERY_CATEGORY in "all" "nota" "non-nota"; do
			for POS in "v" "n" "all"; do
				echo "eval \"\$(conda shell.bash hook)\"; conda activate sembre; python ontonotes_main.py --metric cosine --query-n $QUERY_N --bert-layers $LAYER --query-category $QUERY_CATEGORY --pos $POS" >> $FILENAME
			done
		done
	done
done
cat $FILENAME | xargs -I CMD --max-procs=12 bash -c CMD
