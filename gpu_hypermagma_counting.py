"""
GPU Hypermagma Counting Revolution
====================================
Extends counting invariants to MULTI-VALUED operations.

A hypermagma is a set S with operation * : S x S -> P+(S)
where P+(S) = non-empty subsets of S.

This is strictly MORE general than magmas (which require |a*b| = 1).
Key question: do counting invariants still amplify classification?

n=2: 3^4 = 81 hypermagmas  (non-empty subsets of {0,1}: {{0},{1},{0,1}})
n=3: 7^9 = 40,353,607 hypermagmas (feasible on GPU)
"""

import torch
import time
from itertools import permutations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ==============================================================
# 1. ENCODING: hypermagma as bitmask tensor
# ==============================================================
# For n elements, each product a*b is a subset of {0,...,n-1}
# Encode as n-bit integer: bit k set iff k in a*b
# Operation table: H[a, b] = bitmask in {1, ..., 2^n - 1}  (non-empty only)
# H has shape [B, n, n] with dtype int32

def enumerate_hypermagmas_n2():
    """All 3^4 = 81 hypermagmas on {0,1}. Returns [81, 2, 2] tensor."""
    n = 2
    # Non-empty subsets: 1={0}, 2={1}, 3={0,1}
    num_subsets = 2**n - 1  # 3
    total = num_subsets ** (n * n)  # 3^4 = 81

    indices = torch.arange(total, device=DEVICE, dtype=torch.int32)
    H = torch.zeros(total, n, n, dtype=torch.int32, device=DEVICE)

    idx = indices.clone()
    for row in range(n):
        for col in range(n):
            H[:, row, col] = (idx % num_subsets) + 1  # +1 so 0 is excluded
            idx = idx // num_subsets

    return H


def enumerate_hypermagmas_n3_batch(start, batch_size):
    """
    Generate a batch of hypermagmas on {0,1,2}.
    Non-empty subsets: 7 (bitmasks 1..7)
    """
    n = 3
    num_subsets = 2**n - 1  # 7

    indices = torch.arange(start, start + batch_size, device=DEVICE, dtype=torch.int64)
    H = torch.zeros(batch_size, n, n, dtype=torch.int32, device=DEVICE)

    idx = indices.clone()
    for row in range(n):
        for col in range(n):
            H[:, row, col] = (idx % num_subsets).to(torch.int32) + 1
            idx = idx // num_subsets

    return H


# ==============================================================
# 2. COUNTING INVARIANTS FOR HYPERMAGMAS
# ==============================================================

def compute_hypermagma_invariants(H: torch.Tensor) -> dict:
    """
    H: [B, n, n] int32 bitmask
    Returns dict of [B] float tensors
    """
    B, n, _ = H.shape
    full_mask = (2**n - 1)  # bitmask for S itself

    inv = {}

    # 1. Singleton count: #{(a,b): |a*b|=1} — deterministic products
    popcount = H.int().detach()
    # popcount each element: count bits set
    bit_counts = torch.zeros(B, n, n, device=DEVICE, dtype=torch.int32)
    for bit in range(n):
        bit_counts += ((H >> bit) & 1)

    inv['singleton'] = (bit_counts == 1).sum(dim=(1,2)).float()
    inv['doubleton'] = (bit_counts == 2).sum(dim=(1,2)).float() if n >= 2 else torch.zeros(B, device=DEVICE)
    inv['full_set']  = (bit_counts == n).sum(dim=(1,2)).float()

    # 2. Average product set size (total bits set / n^2)
    inv['total_bits'] = bit_counts.sum(dim=(1,2)).float()

    # 3. Commutative pairs: H[a,b] == H[b,a]
    inv['comm'] = (H == H.transpose(1, 2)).sum(dim=(1,2)).float()

    # 4. Self-membership: a in a*a  (bit a set in H[a,a])
    self_prod = H[:, torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)]  # [B, n]
    elem_bit = (1 << torch.arange(n, device=DEVICE)).unsqueeze(0)  # [1, n]
    inv['self_member'] = ((self_prod & elem_bit) > 0).sum(dim=1).float()

    # 5. "Idempotent subset": H[a,a] == {a}
    singleton_a = (1 << torch.arange(n, device=DEVICE)).to(torch.int32).unsqueeze(0)  # [1, n]
    inv['idemp'] = (self_prod == singleton_a).sum(dim=1).float()

    # 6. Left absorbing: H[a,b] == H_full for all b (a maps everything to S)
    full_rows = (H == full_mask).all(dim=2)  # [B, n] — row a is all full_mask
    inv['left_absorb'] = full_rows.sum(dim=1).float()

    # 7. Right absorbing: H[a,b] == full for all a
    full_cols = (H == full_mask).all(dim=1)  # [B, n]
    inv['right_absorb'] = full_cols.sum(dim=1).float()

    # 8. Left singleton rows: H[a,:] all singletons (left-deterministic)
    left_det = (bit_counts == 1).all(dim=2)  # [B, n]
    inv['left_det_rows'] = left_det.sum(dim=1).float()

    # 9. Kernel size: #{(a,b): H[a,b] contains 0}
    inv['contains_zero'] = ((H & 1) > 0).sum(dim=(1,2)).float()

    # 10. Union coverage: does union of all products = S?
    union_all = H[:, :, :].view(B, -1)
    union = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    for j in range(union_all.shape[1]):
        union = union | union_all[:, j]
    inv['union_coverage'] = (union == full_mask).float()

    return inv


# ==============================================================
# 3. BOOLEAN CLASSIFICATION
# ==============================================================

def boolean_class(H: torch.Tensor) -> torch.Tensor:
    """Returns [B, 5] bool features."""
    B, n, _ = H.shape
    full_mask = (2**n - 1)

    # Is it a classic magma? (all products are singletons)
    bit_counts = torch.zeros(B, n, n, device=DEVICE, dtype=torch.int32)
    for bit in range(n):
        bit_counts += ((H >> bit) & 1)
    is_classic = (bit_counts == 1).all(dim=(1,2))

    # Commutative
    is_comm = (H == H.transpose(1,2)).all(dim=(1,2))

    # Has full-set products
    has_full = (H == full_mask).any(dim=(1,2))

    # Self-member for all
    self_prod = H[:, torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)]
    elem_bit = (1 << torch.arange(n, device=DEVICE)).to(torch.int32).unsqueeze(0)
    all_self_member = ((self_prod & elem_bit) > 0).all(dim=1)

    # All idempotent (H[a,a] = {a})
    singleton_a = (1 << torch.arange(n, device=DEVICE)).to(torch.int32).unsqueeze(0)
    all_idemp = (self_prod == singleton_a).all(dim=1)

    return torch.stack([is_classic, is_comm, has_full, all_self_member, all_idemp], dim=1)


# ==============================================================
# 4. ISOMORPHISM (canonical form)
# ==============================================================

def canonical_form_hyper_cpu(H_np):
    """
    H_np: n x n int array of bitmasks.
    Canonical form under element relabeling.
    """
    n = H_np.shape[0]
    best = None
    for perm in permutations(range(n)):
        p = list(perm)
        p_inv = [0] * n
        for i, v in enumerate(p):
            p_inv[v] = i

        # New table: new_H[i,j] = relabel(H[p[i], p[j]])
        new_H = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                old_mask = H_np[p[i], p[j]]
                # Relabel bits: bit k -> bit p_inv[k]
                new_mask = 0
                for k in range(n):
                    if (old_mask >> k) & 1:
                        new_mask |= (1 << p_inv[k])
                new_H[i][j] = new_mask

        flat = tuple(new_H[i][j] for i in range(n) for j in range(n))
        if best is None or flat < best:
            best = flat
    return best


# ==============================================================
# 5. n=2 EXHAUSTIVE
# ==============================================================

def run_n2():
    print("\n" + "="*60)
    print("n=2 HYPERMAGMA: EXHAUSTIVE (81 total)")
    print("="*60)

    H = enumerate_hypermagmas_n2()  # [81, 2, 2]

    # Iso classification
    H_cpu = H.cpu().numpy()
    iso_map = {}
    iso_ids = []
    for i in range(H.shape[0]):
        cf = canonical_form_hyper_cpu(H_cpu[i])
        if cf not in iso_map:
            iso_map[cf] = len(iso_map)
        iso_ids.append(iso_map[cf])

    n_iso = len(iso_map)
    print(f"Total hypermagmas: 81")
    print(f"Iso classes:       {n_iso}")

    # Boolean
    bool_feats = boolean_class(H)
    bool_tuples = [tuple(bool_feats[i].cpu().tolist()) for i in range(81)]
    iso_to_bool = {}
    for iso_id, bt in zip(iso_ids, bool_tuples):
        iso_to_bool[iso_id] = bt
    n_bool = len(set(iso_to_bool.values()))

    # Counting
    inv = compute_hypermagma_invariants(H)
    iso_to_count = {}
    for i, iso_id in enumerate(iso_ids):
        ct = tuple(inv[k][i].item() for k in sorted(inv.keys()))
        iso_to_count[iso_id] = ct
    n_count = len(set(iso_to_count.values()))

    print(f"Boolean classes:   {n_bool}")
    print(f"Counting classes:  {n_count}")
    print(f"Amplification:     {n_count/n_bool:.1f}x")

    return n_bool, n_count, n_iso


# ==============================================================
# 6. n=3 GPU ENUMERATION
# ==============================================================

def run_n3():
    print("\n" + "="*60)
    print("n=3 HYPERMAGMA: GPU ENUMERATION (40,353,607 total)")
    print("="*60)

    n = 3
    total = (2**n - 1) ** (n*n)  # 7^9 = 40,353,607
    print(f"Total: {total:,}")

    BATCH = 65536
    iso_map = {}
    iso_count_map = {}
    iso_bool_map = {}

    t0 = time.time()
    processed = 0

    for start in range(0, total, BATCH):
        actual_batch = min(BATCH, total - start)
        H = enumerate_hypermagmas_n3_batch(start, actual_batch)

        # Compute invariants
        inv = compute_hypermagma_invariants(H)
        bool_feats = boolean_class(H)

        # Move ALL invariants to CPU at once (avoid per-element .item() calls)
        count_keys = sorted(inv.keys())
        inv_cpu = torch.stack([inv[k] for k in count_keys], dim=1).cpu().numpy()  # [B, K]
        bool_cpu = bool_feats.cpu().numpy()  # [B, 5]

        for b in range(actual_batch):
            ct = tuple(inv_cpu[b].tolist())
            bt = tuple(bool_cpu[b].tolist())
            if ct not in iso_count_map:
                iso_count_map[ct] = len(iso_count_map)
            if bt not in iso_bool_map:
                iso_bool_map[bt] = len(iso_bool_map)

        processed += actual_batch
        if processed % 5_000_000 == 0 or processed == total:
            elapsed = time.time() - t0
            rate = processed / elapsed / 1e6
            print(f"  {processed:>12,} / {total:,}  ({100*processed/total:.1f}%)  "
                  f"{rate:.1f}M/sec  count_classes={len(iso_count_map)}")

    t1 = time.time()
    elapsed = t1 - t0

    n_count = len(iso_count_map)
    n_bool = len(iso_bool_map)

    print(f"\nResults:")
    print(f"Throughput:        {total/elapsed/1e6:.1f}M hypermagmas/sec")
    print(f"Boolean classes:   {n_bool}")
    print(f"Counting classes:  {n_count}")
    print(f"Amplification:     {n_count/n_bool:.1f}x  (lower bound — counting tuples, not iso classes)")
    print(f"NOTE: True iso class count needs canonical form computation")
    print(f"      Counting classes >= Iso classes (multiple iso classes may share a counting tuple)")

    return n_bool, n_count


# ==============================================================
# 7. COMPARISON TABLE
# ==============================================================

def print_comparison(n2_bool, n2_count, n2_iso, n3_bool, n3_count):
    print("\n" + "="*60)
    print("HYPERMAGMA COUNTING REVOLUTION SUMMARY")
    print("="*60)
    print(f"\nExtension of the Counting Revolution to multi-valued operations:\n")
    print(f"{'Structure':<28} {'Bool':>6} {'Count':>8} {'Amp':>6}")
    print("-"*52)
    print(f"{'Regular magmas (n=3)':<28} {'114':>6} {'3,328':>8} {'29x':>6}")
    print(f"{'Hypermagmas n=2 (exact)':<28} {n2_bool:>6} {n2_count:>8} {n2_count/n2_bool:.1f}x")
    print(f"{'Hypermagmas n=3 (lower bnd)':<28} {n3_bool:>6} {n3_count:>8} {n3_count/n3_bool:.1f}x")
    print("-"*52)
    print(f"\nHypermagmas strictly generalize magmas:")
    print(f"  n=2 iso classes: {n2_iso} (vs 10 for regular magmas)")
    print(f"\nCounting invariants:")
    for i, k in enumerate(["singleton", "doubleton", "full_set", "total_bits",
                            "comm", "self_member", "idemp",
                            "left_absorb", "right_absorb", "left_det_rows",
                            "contains_zero", "union_coverage"]):
        print(f"  {i+1:>2}. {k}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    # n=2 exact
    n2_bool, n2_count, n2_iso = run_n2()

    # n=3 GPU
    n3_bool, n3_count = run_n3()

    # Summary
    print_comparison(n2_bool, n2_count, n2_iso, n3_bool, n3_count)

    print("\nDone.")
