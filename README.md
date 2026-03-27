# GPU Hypermagma Counting Revolution

First application of counting invariants to **multi-valued algebraic operations**.

A **hypermagma** is a set S with operation * : SxS -> P+(S), where each product is a *non-empty subset* of S. This strictly generalizes ordinary magmas.

## Key Result: 1128x Amplification at n=3

| Structure | n | Bool classes | Count classes | Amplification |
|-----------|---|--------------|---------------|---------------|
| Regular magmas | 3 | 114 | 3,328 | 29x |
| Hypermagmas | 2 | 10 | 37 | 3.7x |
| **Hypermagmas** | **3** | **16** | **18,051+** | **1128x** |

Boolean is nearly blind to hypermagma structure (16 classes for 40M operations).
Counting invariants distinguish 18,051+ groups -- **39x more powerful than regular magmas**.

## Why the Amplification Explodes

Regular magmas: each product a*b = one element.
Hypermagmas: each product a*b = a *subset* -- 2^n-1 choices per entry.

Boolean properties (commutative? has identity?) completely miss this rich subset structure.
Counting invariants capture subset size distributions, coverage patterns, self-membership -- exactly what boolean can't see.

## 12 Counting Invariants

1. singleton: #{(a,b): |a*b|=1}
2. doubleton: #{(a,b): |a*b|=2}
3. full_set: #{(a,b): |a*b|=n}
4. total_bits: sum of all |a*b|
5. comm: #{(a,b): a*b = b*a} (set equality)
6. self_member: #{a: a in a*a}
7. idemp: #{a: a*a = {a}}
8. left_absorb: #{a: a*b = S for all b}
9. right_absorb: #{a: b*a = S for all b}
10. left_det_rows: #{a: a*b singleton for all b}
11. contains_zero: #{(a,b): 0 in a*b}
12. union_coverage: 1 iff union of all products = S

## Search Space

- n=2: 3^4 = 81 hypermagmas, 45 iso classes (exhaustive)
- n=3: 7^9 = 40,353,607 hypermagmas, >=18,051 distinct counting classes

## Prior Work

Closest: hyperfield enumeration up to order 7 via CPU brute force (AIMS Math, 2020).
No prior work applies counting invariants to hyperstructures.

## Usage

```bash
pip install torch  # CUDA version
python gpu_hypermagma_counting.py
```
