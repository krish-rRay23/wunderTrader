"""Build and verify True LOB Permutation matrix P_true (112 -> 36)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch

def build_true_lob_matrix():
    """Builds projection matrix P_true of shape (112, 36) implementing
    exact depth-ranked order book microstructure signals using the decoded permutation pi.
    """
    cols = []
    names = []
    
    # --- Instrument 0 Inside Quotes & Spreads ---
    # L1 Best Bid = Col 10, L1 Best Ask = Col 14
    # 1. Inside Spread L1: p14 - p10
    v = np.zeros(112, dtype=np.float32); v[14] = 1.0; v[10] = -1.0
    cols.append(v); names.append("i0_Spread_L1")
    
    # 2. Inside Mid-Price: 0.5 * (p10 + p14)
    v = np.zeros(112, dtype=np.float32); v[10] = 0.5; v[14] = 0.5
    cols.append(v); names.append("i0_Mid_L1")
    
    # --- Instrument 1 Inside Quotes & Cross-Asset Basis ---
    # L1 Best Bid i1 = Col 62, L1 Best Ask i1 = Col 66
    # 3. Inside Spread i1: p66 - p62
    v = np.zeros(112, dtype=np.float32); v[66] = 1.0; v[62] = -1.0
    cols.append(v); names.append("i1_Spread_L1")
    
    # 4. Cross-Asset Mid Basis: Mid(i0) - Mid(i1)
    v = np.zeros(112, dtype=np.float32)
    v[10] = 0.5; v[14] = 0.5
    v[62] = -0.5; v[66] = -0.5
    cols.append(v); names.append("CrossAsset_MidBasis")
    
    # --- Depth-Ranked Volume Imbalances (Bids vs Asks) ---
    # Bid Volume Cols: L1=32, L2=27, L3=24, L4=23, L5=29, L6=26, L7=28, L8=31, L9=30, L10=22, L11=25
    # Ask Volume Cols: L1=36, L2=43, L3=33, L4=37, L5=39, L6=34, L7=35, L8=40, L9=42, L10=38, L11=41
    bid_v_cols = [32, 27, 24, 23, 29, 26, 28, 31, 30, 22, 25]
    ask_v_cols = [36, 43, 33, 37, 39, 34, 35, 40, 42, 38, 41]
    
    # 5. L1 Volume Imbalance: v_bid1 - v_ask1
    v = np.zeros(112, dtype=np.float32); v[32] = 1.0; v[36] = -1.0
    cols.append(v); names.append("i0_VolImb_L1")
    
    # 6. L2 Volume Imbalance: v_bid2 - v_ask2
    v = np.zeros(112, dtype=np.float32); v[27] = 1.0; v[43] = -1.0
    cols.append(v); names.append("i0_VolImb_L2")
    
    # 7. L3 Volume Imbalance: v_bid3 - v_ask3
    v = np.zeros(112, dtype=np.float32); v[24] = 1.0; v[33] = -1.0
    cols.append(v); names.append("i0_VolImb_L3")
    
    # 8. Top-3 Decayed Volume Imbalance (Stoikov 2018):
    v = np.zeros(112, dtype=np.float32)
    v[32] = 1.00; v[27] = 0.70; v[24] = 0.49
    v[36] = -1.00; v[43] = -0.70; v[33] = -0.49
    cols.append(v); names.append("i0_Decayed_Top3_Imb")
    
    # 9. Top-5 Decayed Volume Imbalance:
    v = np.zeros(112, dtype=np.float32)
    for lvl in range(5):
        w_l = float(np.exp(-0.35 * lvl))
        v[bid_v_cols[lvl]] = w_l
        v[ask_v_cols[lvl]] = -w_l
    cols.append(v); names.append("i0_Decayed_Top5_Imb")
    
    # 10. All-11 DIMI (Decayed Integrated Micro-Price Imbalance):
    v = np.zeros(112, dtype=np.float32)
    for lvl in range(11):
        w_l = float(np.exp(-0.35 * lvl))
        v[bid_v_cols[lvl]] = w_l
        v[ask_v_cols[lvl]] = -w_l
    cols.append(v); names.append("i0_DIMI_All11")
    
    # --- Instrument 1 Depth Imbalances ---
    # i1 Bid Vol Cols: L1=84, L2=79, L3=76
    # i1 Ask Vol Cols: L1=88, L2=95, L3=85
    # 11. i1 L1 Volume Imbalance:
    v = np.zeros(112, dtype=np.float32); v[84] = 1.0; v[88] = -1.0
    cols.append(v); names.append("i1_VolImb_L1")
    
    # 12. i1 Top-3 Decayed Volume Imbalance:
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
        
    # Trades i1_dp0..dp3: Cols 96..99 relative to i1 Mid
    for k in range(2):
        v = np.zeros(112, dtype=np.float32)
        v[96 + k] = 1.0; v[62] = -0.5; v[66] = -0.5
        cols.append(v); names.append(f"i1_TradeDisp_{k}")
        
    # --- Trade Volume Flows ---
    # i0_dv0 - dv1: Cols 48, 49
    v = np.zeros(112, dtype=np.float32); v[48] = 1.0; v[49] = -1.0
    cols.append(v); names.append("i0_TradeVolFlow_01")
    
    # Cross-Instrument Taker Flow: dv0(i1) - dv0(i0): Col 100 - Col 48
    v = np.zeros(112, dtype=np.float32); v[100] = 1.0; v[48] = -1.0
    cols.append(v); names.append("CrossTradeVolFlow")
    
    # --- Latent Alpha Features a0..a7 (Cols 104..111) ---
    for a_idx in range(8):
        v = np.zeros(112, dtype=np.float32)
        v[104 + a_idx] = 1.0
        cols.append(v); names.append(f"Alpha_{a_idx}")
        
    # --- Selected High-Relevance Raw Price Features ---
    # cols 3, 0, 8, 7, 2, 6, 9
    for raw_c in [3, 0, 8, 7, 2, 6, 9, 13]:
        v = np.zeros(112, dtype=np.float32)
        v[raw_c] = 1.0
        cols.append(v); names.append(f"RawPrice_{raw_c}")
        
    P = np.column_stack(cols) # shape (112, 36)
    return P, names

if __name__ == "__main__":
    P, names = build_true_lob_matrix()
    print(f"Built True LOB Matrix: shape {P.shape} with {len(names)} features:")
    for idx, name in enumerate(names):
        nonzeros = np.where(P[:, idx] != 0)[0]
        print(f"  {idx:2d}. {name:<24} | Nonzero inputs: {nonzeros.tolist()}")
