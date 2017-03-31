if [ "$#" -ne 3 ]; then
    echo "Usage: bash predict_probs.sh $images_glob $output_path $gpu_index"
fi

export PATH="/import/local/ssd1/jschoenb/dev/code/miniconda3/bin:$PATH"
export PYTHONPATH="/import/local/ssd1/jschoenb/dev/python:$PYTHONPATH"
source activate dilation
mkdir -p $2
python predict_probs.py \
    --dataset cityscapes \
    --input_glob "$1" \
    --output_path "$2" \
    --gpu $3
