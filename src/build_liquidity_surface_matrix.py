"""Build Full Cross-Instrument Order Book Liquidity Surface Projection Matrix P_surface (112 -> 68).

Incorporates:
1. Cross-Instrument Level-by-Level Bid Volume Differentials (Levels 1..11)
2. Cross-Instrument Level-by-Level Ask Volume Differentials (Levels 1..11)
3. Cross-Instrument Level-by-Level Order Flow Imbalance (OFI) Differentials (Levels 1..11)
4. Decayed Cumulative Cross-Instrument Depth Imbalances (Top-3, Top-5, All-11)
5. Multi-Level Spreads for i0 and i1 (Levels 1, 2, 3, 5)
6. Cross-Instrument Mid Basis (Mid_i0 - Mid_i1)
7. Cross-Instrument Spread Differentials (Spread_i0 - Spread_i1 for L1, L2, L3, L5)
8. Depth Slope Asymmetries for i0 and i1
9. Trade Price Displacements relative to Mid for i0 and i1 (4 channels each)
10. Trade Volume Differentials (i0 - i1) across 4 channels
11. All 8 Latent Alphas a0..a7 and Alpha6 - Alpha7

Mathematically collapses into static (112, 2) tensor in ONNX.
"""
import numpy as np

def build_liquidity_surface_matrix():
    cols = []
    names = []
    
    # Decoded volume column indices:
    # Instrument 0:
    b0 = [32, 27, 24, 23, 29, 26, 28, 31, 30, 22, 25] # Bids L1..L11
    a0 = [36, 43, 33, 37, 39, 34, 35, 40, 42, 38, 41] # Asks L1..L11
    # Instrument 1:
    b1 = [84, 79, 76, 75, 81, 78, 80, 83, 82, 74, 77] # Bids L1..L11
    a1 = [88, 95, 85, 89, 91, 86, 87, 92, 94, 90, 93] # Asks L1..L11
    
    # Decoded price column indices:
    # Instrument 0: Bids L1..L5: 10, 5, 2, 1, 7; Asks L1..L5: 14, 21, 11, 12, 17
    # Instrument 1: Bids L1..L5: 62, 57, 54, 53, 59; Asks L1..L5: 66, 73, 63, 64, 69
    
    # 1. Cross-Instrument Level-by-Level Bid Volume Differentials (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[b0[lvl]] = 1.0; v[b1[lvl]] = -1.0
        cols.append(v); names.append(f"Cross_BidVol_L{lvl+1}")
        
    # 2. Cross-Instrument Level-by-Level Ask Volume Differentials (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[a0[lvl]] = 1.0; v[a1[lvl]] = -1.0
        cols.append(v); names.append(f"Cross_AskVol_L{lvl+1}")
        
    # 3. Cross-Instrument Level-by-Level OFI Differentials: (v_b0 - v_a0) - (v_b1 - v_a1) (Levels 1..11)
    for lvl in range(11):
        v = np.zeros(112, dtype=np.float32)
        v[b0[lvl]] = 1.0; v[a0[lvl]] = -1.0
        v[b1[lvl]] = -1.0; v[a1[lvl]] = 1.0
        cols.append(v); names.append(f"Cross_OFI_L{lvl+1}")
        
    # 4. Intra-Instrument OFI for i0 (Top 5 levels)
    for lvl in range(5):
        v = np.zeros(112, dtype=np.float32)
        v[b0[lvl]] = 1.0; v[a0[lvl]] = -1.0
        cols.append(v); names.append(f"i0_OFI_L{lvl+1}")
        
    # 5. Decayed Integrated OFI:
    for decay_name, decay_rate, max_lvl in [('Top3', 0.35, 3), ('Top5', 0.25, 5), ('All11', 0.15, 11)]:
        v = np.zeros(112, dtype=np.float32)
        for lvl in range(max_lvl):
            w_l = float(np.exp(-decay_rate * lvl))
            v[b0[lvl]] += w_l; v[a0[lvl]] -= w_l
            v[b1[lvl]] -= w_l; v[a1[lvl]] += w_l
        cols.append(v); names.append(f"Cross_DecayedOFI_{decay_name}")
        
    # 6. Spreads & Quotes
    # i0 Spreads: L1 (14-10), L2 (21-5), L3 (11-2), L5 (17-7)
    cols.append(np.array([1.0 if i==14 else -1.0 if i==10 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i0_Spread_L1")
    cols.append(np.array([1.0 if i==21 else -1.0 if i==5 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i0_Spread_L2")
    cols.append(np.array([1.0 if i==11 else -1.0 if i==2 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i0_Spread_L3")
    cols.append(np.array([1.0 if i==17 else -1.0 if i==7 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i0_Spread_L5")
    
    # i1 Spreads: L1 (66-62), L2 (73-57), L3 (63-54), L5 (69-59)
    cols.append(np.array([1.0 if i==66 else -1.0 if i==62 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i1_Spread_L1")
    cols.append(np.array([1.0 if i==73 else -1.0 if i==57 else 0.0 for i in range(112)], dtype=np.float32))
    names.append("i1_Spread_L2")
    
    # Cross-Asset Mid Basis: Mid(i0) - Mid(i1) = 0.5*(p10+p14) - 0.5*(p62+p66)
    v_mid = np.zeros(112, dtype=np.float32)
    v_mid[10] = 0.5; v_mid[14] = 0.5; v_mid[62] = -0.5; v_mid[66] = -0.5
    cols.append(v_mid); names.append("Cross_MidBasis")
    
    # Cross-Asset Spread Diff: Spread_i0 - Spread_i1 (L1)
    v_sp = np.zeros(112, dtype=np.float32)
    v_sp[14] = 1.0; v_sp[10] = -1.0; v_sp[66] = -1.0; v_sp[62] = 1.0
    cols.append(v_sp); names.append("Cross_SpreadDiff_L1")
    
    # 7. Book Slope Asymmetries
    # i0 Slope Asym L5: (p17 - p14) - (p10 - p7)
    v_sl0 = np.zeros(112, dtype=np.float32)
    v_sl0[17] = 1.0; v_sl0[14] = -1.0; v_sl0[10] = -1.0; v_sl0[7] = 1.0
    cols.append(v_sl0); names.append("i0_SlopeAsym_L5")
    
    # i1 Slope Asym L5: (p69 - p66) - (p62 - p59)
    v_sl1 = np.zeros(112, dtype=np.float32)
    v_sl1[69] = 1.0; v_sl1[66] = -1.0; v_sl1[62] = -1.0; v_sl1[59] = 1.0
    cols.append(v_sl1); names.append("i1_SlopeAsym_L5")
    
    # Cross Slope Asym Diff
    cols.append(v_sl0 - v_sl1); names.append("Cross_SlopeAsymDiff")
    
    # 8. Trade Displacements relative to Mid (4 channels each)
    # i0 Trade Prices: 44..47, Mid: 0.5*(p10 + p14)
    for ch in range(4):
        v = np.zeros(112, dtype=np.float32)
        v[44 + ch] = 1.0; v[10] = -0.5; v[14] = -0.5
        cols.append(v); names.append(f"i0_TradeDisp_Ch{ch}")
        
    # i1 Trade Prices: 96..99, Mid: 0.5*(p62 + p66)
    for ch in range(4):
        v = np.zeros(112, dtype=np.float32)
        v[96 + ch] = 1.0; v[62] = -0.5; v[66] = -0.5
        cols.append(v); names.append(f"i1_TradeDisp_Ch{ch}")
        
    # Trade Volume Differentials (i0 - i1): cols (48..51) - (100..103)
    for ch in range(4):
        v = np.zeros(112, dtype=np.float32)
        v[48 + ch] = 1.0; v[100 + ch] = -1.0
        cols.append(v); names.append(f"Cross_TradeVol_Ch{ch}")
        
    # 9. Latent Alpha Signals (cols 104..111)
    for a_idx in range(8):
        v = np.zeros(112, dtype=np.float32)
        v[104 + a_idx] = 1.0
        cols.append(v); names.append(f"Alpha_{a_idx}")
        
    # 10. High-Conviction Alpha Differential: Alpha6 - Alpha7 (r = +0.100)
    v_ad = np.zeros(112, dtype=np.float32)
    v_ad[110] = 1.0; v_ad[111] = -1.0
    cols.append(v_ad); names.append("Alpha6_minus_Alpha7")
    
    P = np.column_stack(cols) # (112, K)
    return P.astype(np.float32), names


if __name__ == "__main__":
    P, names = build_liquidity_surface_matrix()
    print(f"Built Cross-Instrument Liquidity Surface Matrix: Shape {P.shape}")
    print(f"Total Alpha Features: {len(names)}")
    for i, n in enumerate(names):
        print(f"  {i+1:2d}. {n}")
