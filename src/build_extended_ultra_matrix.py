"""Build Extended Ultra LOB Matrix (112 -> 81).

Features 0..46: Exact 47 signals from build_ultra_lob_matrix (preserving the 0.6517 champion manifold).
Features 47..57: Cross-Instrument Bid Volume Differentials (Levels 1..11)
Features 58..68: Cross-Instrument Ask Volume Differentials (Levels 1..11)
Features 69..79: Cross-Instrument OFI Differentials (Levels 1..11)
Feature 80: Alpha6 - Alpha7 (r = +0.100)

This allows zero-loss warm-starting from best_ultra_connectome.pt (weights for 0..46 loaded directly,
47..80 initialized to zeros). Step 0 score is guaranteed to match 0.427712.
"""
import sys
sys.path.insert(0, ".")
import numpy as np

from src.build_ultra_lob_matrix import build_ultra_lob_matrix

def build_extended_ultra_matrix():
    P_base, names_base = build_ultra_lob_matrix() # (112, 47)
    cols = [P_base[:, i] for i in range(P_base.shape[1])]
    names = list(names_base)
    
    # Decoded volume columns:
    b0 = [32, 27, 24, 23, 29, 26, 28, 31, 30, 22, 25] # Bids i0 L1..L11
    a0 = [36, 43, 33, 37, 39, 34, 35, 40, 42, 38, 41] # Asks i0 L1..L11
    b1 = [84, 79, 76, 75, 81, 78, 80, 83, 82, 74, 77] # Bids i1 L1..L11
    a1 = [88, 95, 85, 89, 91, 86, 87, 92, 94, 90, 93] # Asks i1 L1..L11
    
    # 1. Cross-Instrument Bid Volumes: v_b0 - v_b1 (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[b0[lvl]] = 1.0; v[b1[lvl]] = -1.0
        cols.append(v); names.append(f"Cross_BidVol_L{lvl+1}")
        
    # 2. Cross-Instrument Ask Volumes: v_a0 - v_a1 (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[a0[lvl]] = 1.0; v[a1[lvl]] = -1.0
        cols.append(v); names.append(f"Cross_AskVol_L{lvl+1}")
        
    # 3. Cross-Instrument OFI: (v_b0 - v_a0) - (v_b1 - v_a1) (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[b0[lvl]] = 1.0; v[a0[lvl]] = -1.0
        v[b1[lvl]] = -1.0; v[a1[lvl]] = 1.0
        cols.append(v); names.append(f"Cross_OFI_L{lvl+1}")
        
    # 4. Alpha6 - Alpha7
    v_a = np.zeros(112, dtype=np.float32)
    v_a[110] = 1.0; v_a[111] = -1.0
    cols.append(v_a); names.append("Alpha6_minus_Alpha7")
    
    P_ext = np.column_stack(cols).astype(np.float32)
    return P_ext, names


if __name__ == "__main__":
    P, names = build_extended_ultra_matrix()
    print(f"Built Extended Ultra Matrix: Shape {P.shape} ({len(names)} features)")
    print(f"Features 0..46: Champion Ultra Base")
    print(f"Features 47..{len(names)-1}: New Cross-Instrument Liquidity Extensions")
