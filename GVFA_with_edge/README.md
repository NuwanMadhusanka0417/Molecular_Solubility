# Ridge baseline (unchanged)
python GVFA_edge_main.py --model gvfa_ridge

# Attention model (all fixes applied)
python GVFA_edge_main.py --model attn_gvfa --epochs 200 --patience 20



Thos code work perfect for new (GNN paper) dataset. for datset parameter, use solubility_novel