

# 1. invert
python invert.py --config configs/retriever.yaml 


# 2. sample
python run.py --config configs/retriever.yaml


# 2-1. (optional) apply reframing
python run.py --config configs/retriever.yaml --transformation "shift"  --shift_dir "left" 

python run.py --config configs/retriever.yaml --transformation "resize" --resize_factor 2.0  

python run.py --config configs/retriever.yaml --transformation "warp"   --shift_dir "right"