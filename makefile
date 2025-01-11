all: 
	clear 
	CUDA_VISIBLE_DEVICES=7 python3 src/main.py --config=dmcg --env-config=gather with cg_edges=full seed=10 
