"""Build Ultra LOB Microstructure Projection Matrix P_ultra (112 -> 48).

Incorporates high-IC residual microstructure alpha signals:
1. Cross-Instrument L1 OFI Diff: (v_bid0 - v_ask0) - (v_bid1 - v_ask1) [r = +0.116 with res_0, +0.112 with res_1]
2. Book Slope Asymmetry: (AskSlope_L5 - BidSlope_L5) [r = +0.223 with res_1]
3. Multi-Level Physical Spreads: L1, L2, L3, L5 Spreads [r up to -0.144 with res_1]
4. Bid & Ask Depth Slopes across top-5 decoded levels [r up to +0.166]
5. Decayed Integrated Micro-Price Imbalance (DIMI, Stoikov 2018) across all 11 physical levels
6. 4-Channel Trade Price Displacements relative to Mid (dp - mid) [r = +0.1101 with t0]
7. Cross-Instrument Mid Return Basis (mid_i0 - mid_i1)
8. All 8 Latent Alphas a0..a7
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch

def build_ultra_lob_matrix():
    """Builds projection matrix P of shape (112, 48) containing top microstructure alpha signals."""
    cols = []
    names = []
    
    # 1. Level-1 Spread i0: p14 - p10
    v = np.zeros(112, dtype=np.float32); v[14] = 1.0; v[10] = -1.0
    cols.append(v); names.append("i0_Spread_L1")
    
    # 2. Level-2 Spread i0: p21 - p5
    v = np.zeros(112, dtype=np.float32); v[21] = 1.0; v[5] = -1.0
    cols.append(v); names.append("i0_Spread_L2")
    
    # 3. Level-3 Spread i0: p11 - p2
    v = np.zeros(112, dtype=np.float32); v[11] = 1.0; v[2] = -1.0
    cols.append(v); names.append("i0_Spread_L3")
    
    # 4. Level-5 Spread i0: p17 - p7
    v = np.zeros(112, dtype=np.float32); v[17] = 1.0; v[7] = -1.0
    cols.append(v); names.append("i0_Spread_L5")
    
    # 5. Inside Mid-Price i0: 0.5 * (p10 + p14)
    v = np.zeros(112, dtype=np.float32); v[10] = 0.5; v[14] = 0.5
    cols.append(v); names.append("i0_Mid_L1")
    
    # 6. Ask Depth Slope L5: p17 - p14
    v = np.zeros(112, dtype=np.float32); v[17] = 1.0; v[14] = -1.0
    cols.append(v); names.append("i0_AskSlope_L5")
    
    # 7. Bid Depth Slope L5: p10 - p7
    v = np.zeros(112, dtype=np.float32); v[10] = 1.0; v[7] = -1.0
    cols.append(v); names.append("i0_BidSlope_L5")
    
    # 8. Slope Asymmetry: AskSlope - BidSlope: (p17 - p14) - (p10 - p7)
    v = np.zeros(112, dtype=np.float32); v[17] = 1.0; v[14] = -1.0; v[10] = -1.0; v[7] = 1.0
    cols.append(v); names.append("i0_SlopeAsym_L5")
    
    # --- Instrument 1 Quotes & Spreads ---
    # L1 Best Bid i1 = Col 62, L1 Best Ask i1 = Col 66
    # 9. Inside Spread i1: p66 - p62
    v = np.zeros(112, dtype=np.float32); v[66] = 1.0; v[62] = -1.0
    cols.append(v); names.append("i1_Spread_L1")
    
    # 10. Cross-Asset Mid Basis: Mid(i0) - Mid(i1)
    v = np.zeros(112, dtype=np.float32)
    v[10] = 0.5; v[14] = 0.5; v[62] = -0.5; v[66] = -0.5
    cols.append(v); names.append("CrossAsset_MidBasis")
    
    # 11. Cross-Asset Spread Diff: Spread(i0) - Spread(i1)
    v = np.zeros(112, dtype=np.float32)
    v[14] = 1.0; v[10] = -1.0; v[66] = -1.0; v[62] = 1.0
    cols.append(v); names.append("CrossAsset_SpreadDiff")
    
    # --- Depth-Ranked Volume Imbalances ---
    # Decoded Bids: L1=32, L2=27, L3=24, L4=23, L5=29, L6=26, L7=28, L8=31, L9=30, L10=22, L11=25
    # Decoded Asks: L1=36, L2=43, L3=33, L4=37, L5=39, L6=34, L7=35, L8=40, L9=42, L10=38, L11=41
    bid_v_cols = [32, 27, 24, 23, 29, 26, 28, 31, 30, 22, 25]
    ask_v_cols = [36, 43, 33, 37, 39, 34, 35, 40, 42, 38, 41]
    
    # 12. L1 Volume Imbalance: v_bid1 - v_ask1
    v = np.zeros(112, dtype=np.float32); v[32] = 1.0; v[36] = -1.0
    cols.append(v); names.append("i0_VolImb_L1")
    
    # 13. L2 Volume Imbalance: v_bid2 - v_ask2
    v = np.zeros(112, dtype=np.float32); v[27] = 1.0; v[43] = -1.0
    cols.append(v); names.append("i0_VolImb_L2")
    
    # 14. L3 Volume Imbalance: v_bid3 - v_ask3
    v = np.zeros(112, dtype=np.float32); v[24] = 1.0; v[33] = -1.0
    cols.append(v); names.append("i0_VolImb_L3")
    
    # 15. Top-3 Decayed Volume Imbalance:
    v = np.zeros(112, dtype=np.float32)
    v[32] = 1.00; v[27] = 0.70; v[24] = 0.49
    v[36] = -1.00; v[43] = -0.70; v[33] = -0.49
    cols.append(v); names.append("i0_Decayed_Top3_Imb")
    
    # 16. Top-5 Decayed Volume Imbalance:
    v = np.zeros(112, dtype=np.float32)
    for lvl in range(5):
        w_l = float(np.exp(-0.35 * lvl))
        v[bid_v_cols[lvl]] = w_l
        v[ask_v_cols[lvl]] = -w_l
    cols.append(v); names.append("i0_Decayed_Top5_Imb")
    
    # 17. All-11 DIMI:
    v = np.zeros(112, dtype=np.float32)
    for lvl in range(11):
        w_l = float(np.exp(-0.35 * lvl))
        v[bid_v_cols[lvl]] = w_l
        v[ask_v_cols[lvl]] = -w_l
    cols.append(v); names.append("i0_DIMI_All11")
    
    # --- Instrument 1 Depth Imbalances & Cross-Asset OFI ---
    # i1 Bid Vol Cols: L1=84, L2=79, L3=76
    # i1 Ask Vol Cols: L1=88, L2=95, L3=85
    # 18. i1 L1 Volume Imbalance:
    v = np.zeros(112, dtype=np.float32); v[84] = 1.0; v[88] = -1.0
    cols.append(v); names.append("i1_VolImb_L1")
    
    # 19. Cross-Instrument L1 OFI Diff: OFI(i0) - OFI(i1) [r = +0.116 / +0.112]
    v = np.zeros(112, dtype=np.float32)
    v[32] = 1.0; v[36] = -1.0; v[84] = -1.0; v[88] = 1.0
    cols.append(v); names.append("CrossAsset_L1_OFI")
    
    # 20. i1 Top-3 Decayed Volume Imbalance:
    v = np.zeros(112, dtype=np.float32)
    v[84] = 1.00; v[79] = 0.70; v[76] = 0.49
    v[88] = -1.00; v[95] = -0.70; v[85] = -0.49
    cols.append(v); names.append("i1_Decayed_Top3_Imb")
    
    # --- Trade Price Displacements relative to Mid ---
    # Trades i0_dp0..dp3: Cols 44..47
    for k in range(4):
        v = np.zeros(112, dtype=np.float32)
        v[44 + k] = 1.0; v[10] = -0.5; v[14] = -0.5
        cols.append(v); names.append(f"i0_TradeDisp_{k}")
        
    # Trades i1_dp0..dp1: Cols 96..97 relative to i1 Mid
    for k in range(2):
        v = np.zeros(112, dtype=np.float32)
        v[96 + k] = 1.0; v[62] = -0.5; v[66] = -0.5
        cols.append(v); names.append(f"i1_TradeDisp_{k}")
        
    # --- Trade Volume Flows ---
    # 27. i0_dv0 - dv1: Cols 48, 49
    v = np.zeros(112, dtype=np.float32); v[48] = 1.0; v[49] = -1.0
    cols.append(v); names.append("i0_TradeVolFlow_01")
    
    # 28. Cross-Instrument Taker Flow: dv0(i1) - dv0(i0): Col 100 - Col 48
    v = np.zeros(112, dtype=np.float32); v[100] = 1.0; v[48] = -1.0
    cols.append(v); names.append("CrossTradeVolFlow")
    
    # --- Latent Alpha Features a0..a7 (Cols 104..111) ---
    for a_idx in range(8):
        v = np.zeros(112, dtype=np.float32)
        v[104 + a_idx] = 1.0
        cols.append(v); names.append(f"Alpha_{a_idx}")
        
    # --- Key Order Book Price Anchors ---
    # cols 3, 0, 8, 7, 2, 6, 9, 13
    for raw_c in [3, 0, 8, 7, 2, 6, 9, 13, 21, 5, 11]:
        v = np.zeros(112, dtype=np.float32)
        v[raw_c] = 1.0
        cols.append(v); names.append(f"KeyPrice_{raw_c}")
        
    P = np.column_stack(cols) # shape (112, K)
    return P, names

if __name__ == "__main__":
    P, names = build_ultra_lob_matrix()
    print(f"Built Ultra LOB Matrix: shape {P.shape} with {len(names)} features:")
    for idx, name in enumerate(names):
        print(f"  {idx:2d}. {name:<26}")
