"""
DIAGNOSTIC SCRIPT: Find Why Improvements Aren't Working
========================================================

This will help identify the exact problem.
"""

import torch
from src.create_graphs import create_graph_list
from src.load_data import load_data
from src.VSA_conversion import VSA_conversion

# Load small sample
train_data, test_data = load_data()

# Create graphs
train_graphs = create_graph_list(train_data)[:10]  # Just 10 graphs for debugging

print("="*70)
print("DIAGNOSTIC 1: Check if edge_attr exists and has correct shape")
print("="*70)

for i, g in enumerate(train_graphs[:3]):
    print(f"\nGraph {i}:")
    print(f"  Has edge_attr: {hasattr(g, 'edge_attr')}")
    if hasattr(g, 'edge_attr'):
        print(f"  edge_attr shape: {g.edge_attr.shape}")
        print(f"  edge_attr sample (first 3 edges):\n{g.edge_attr[:3]}")
        print(f"  edge_attr dtype: {g.edge_attr.dtype}")
        
        # Check values
        print(f"  Bond types (col 0): {g.edge_attr[:3, 0]}")
        print(f"  Conjugated (col 1): {g.edge_attr[:3, 1]}")
        print(f"  In ring (col 2): {g.edge_attr[:3, 2]}")
        print(f"  Length (col 3): {g.edge_attr[:3, 3]}")
        print(f"  Stereo (col 4): {g.edge_attr[:3, 4]}")


print("\n" + "="*70)
print("DIAGNOSTIC 2: Check after VSA conversion")
print("="*70)

train_HVs = VSA_conversion(train_graphs.copy(), 1000, projection_type="orthogonal")

for i, g in enumerate(train_HVs[:3]):
    print(f"\nGraph {i} after VSA_conversion:")
    print(f"  Has edge_attr: {hasattr(g, 'edge_attr')}")
    if hasattr(g, 'edge_attr'):
        print(f"  edge_attr still exists: YES")
        print(f"  edge_attr shape: {g.edge_attr.shape}")
    else:
        print(f"  edge_attr still exists: NO - THIS IS THE PROBLEM!")


print("\n" + "="*70)
print("DIAGNOSTIC 3: Test edge strength computation")
print("="*70)

from models.graphcnnVSA_Binding_FULL import GraphCNN

model = GraphCNN(
    1000, 3, 1, 'sum', 'sum', torch.device('cpu'), 12,
    edge_feat_dim=5,
    use_hier_khop=True,
    max_hops=3,
    use_edge_strength=True,
    use_positional_encoding=True,
)

# Manually test edge strength computation
if hasattr(train_HVs[0], 'edge_attr'):
    test_edge_attr = train_HVs[0].edge_attr[:5]  # First 5 edges
    print(f"\nTest edge_attr:\n{test_edge_attr}")
    
    strengths = model._compute_edge_strengths(test_edge_attr)
    print(f"\nComputed strengths:\n{strengths}")
    print(f"Strength range: [{strengths.min():.3f}, {strengths.max():.3f}]")
    
    if torch.allclose(strengths, torch.ones_like(strengths)):
        print("\n❌ PROBLEM: All strengths are 1.0! Edge strength modulation not working!")
    else:
        print("\n✅ Edge strength modulation IS working (strengths vary)")
else:
    print("\n❌ CRITICAL: edge_attr doesn't exist after VSA_conversion!")


print("\n" + "="*70)
print("DIAGNOSTIC 4: Check GraphCNN preprocessing")
print("="*70)

# Forward pass with single batch
batch = train_HVs[:2]

print(f"\nInput batch size: {len(batch)}")
print(f"Graph 0 has edge_attr: {hasattr(batch[0], 'edge_attr')}")
print(f"Graph 1 has edge_attr: {hasattr(batch[1], 'edge_attr')}")

# Run forward to see what happens
output = model(batch)
print(f"\nOutput shape: {output.shape}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)