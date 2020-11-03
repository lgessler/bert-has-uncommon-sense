MODELS=(
	"bert-base-cased"
	"bert-large-cased"
)
METRICS=( "cosine" "euclidean" )

echo "run_all.sh: running main.py"
for MODEL in "${MODELS[@]}"
do 
	for METRIC in "${METRICS[@]}"
	do
		python main.py $MODEL $METRIC
	done
done

echo "run_all.sh: running score.py"
for MODEL in "${MODELS[@]}"
do 
	for METRIC in "${METRICS[@]}"
	do
		python score.py "${METRIC}_predictions/${MODEL}.tsv" &
	done
done


