/*
 * cosine_tree - the cosine normalise's reciprocal norm, one definition for every pass
 * that computes it: `attention.comp`'s row pass and the QKV projection's own epilogue
 * (`qkv_epilogue.glsl`). The two must agree to the bit, and the only way to be sure they
 * do is for there to be one copy of the arithmetic.
 *
 * Needs `half_round` from publish.glsl.
 */

const float COSINE_NORM_FLOOR = 0.00006198883056640625;

/* half(l*r), half(l+r), half(l*r + acc) - each a float32 expression with one rounding
 * at the end, matching `_half_multiply` / `_half_add` / `_half_fma`. `precise` stops an
 * FMA contraction, which would drop the float32 rounding between the multiply and the
 * add and give a different answer. */
float hmul(float l, float r) { precise float t = l * r; return half_round(t); }
float hadd(float l, float r) { precise float t = l + r; return half_round(t); }
float hfma(float l, float r, float acc) { precise float t = l * r + acc; return half_round(t); }

/* `h` is one head's 32 channels, already rounded to half. The vendor's fragment tree:
 * eight partial sums of four squares, then two butterfly levels, then the last add -
 * each step a half operation, in this order and no other. */
float cosine_reciprocal(float h[32]) {
    float partial[4][2];
    for (uint lane = 0u; lane < 4u; lane++)
        for (uint parity = 0u; parity < 2u; parity++) {
            uint ch = lane * 2u + parity;
            float first  = hfma(h[ch + 8u],  h[ch + 8u],  hmul(h[ch],       h[ch]));
            float second = hfma(h[ch + 24u], h[ch + 24u], hmul(h[ch + 16u], h[ch + 16u]));
            partial[lane][parity] = hadd(first, second);
        }
    float two[4][2], one[4][2];
    for (uint lane = 0u; lane < 4u; lane++)
        for (uint p = 0u; p < 2u; p++) two[lane][p] = hadd(partial[lane][p], partial[lane ^ 2u][p]);
    for (uint lane = 0u; lane < 4u; lane++)
        for (uint p = 0u; p < 2u; p++) one[lane][p] = hadd(two[lane][p], two[lane ^ 1u][p]);

    float norm = max(hadd(one[0][0], one[0][1]), half_round(COSINE_NORM_FLOOR));
    return half_round(inversesqrt(norm));
}

/* The ViT's own (OpenDLSS-NR's `vit.wgsl`, notes/opendlss-reference.md): each pair square
 * summed in float32 - the low one exact, the high one rounded to half first - and rounded
 * once, then a stride 8, 4 and 2 tree of half adds, the last two pairs apart, and the
 * reciprocal square root as a divide. The floor keeps a zero head zero, as the vendor's
 * NaN publishing as +0 does. `nr_model._vit_norm`. */
float vit_reciprocal(float h[32]) {
    float r[16];
    for (uint c = 0u; c < 16u; c++) r[c] = hfma(h[c], h[c], hmul(h[c + 16u], h[c + 16u]));
    for (uint c = 0u; c < 8u; c++) r[c] = hadd(r[c], r[c + 8u]);
    for (uint c = 0u; c < 4u; c++) r[c] = hadd(r[c], r[c + 4u]);
    float norm = max(hadd(hadd(r[0], r[2]), hadd(r[1], r[3])), half_round(COSINE_NORM_FLOOR));
    return half_round(1.0 / sqrt(norm));
}

/* half(sqrt(32)): the ViT's query takes it as a half multiply of its own, between the norm
 * and the learned scale. */
const float VIT_ROOT = 5.65625;

/* Both trees spread over four adjacent lanes of a subgroup, for an includer that enables
 * GL_KHR_shader_subgroup_shuffle and defines COSINE_TREE_SHUFFLE. Lane l of the four holds
 * the row's channels g*8 + 2l + p as h[2g + p] (g 0-3, p 0-1), already rounded to half -
 * the fragment layout the cosine tree is written in. Both trees then share one shape: a
 * partial sum per parity from the lane's own channels, the butterflies two and one lanes
 * apart, and the two parities added last.
 *   cosine: partial[l][p] = (c+8)^2 + c^2, then (c+24)^2 + (c+16)^2, added (c = 2l + p);
 *           two = partial[l] + partial[l^2]; one = two[l] + two[l^1]; one[0][0] + one[0][1].
 *   vit:    r[c] = c^2 + (c+16)^2 for c and c+8, added; r[c] + r[c+4] is lanes l and l^2;
 *           (r0 + r2) and (r1 + r3) are parity 0 and 1 across lanes l and l^1.
 * The same half operations on the same operands as cosine_reciprocal and vit_reciprocal;
 * a half add is commutative, so all four lanes end with the same reciprocal. All four must
 * execute it (the shuffles), in uniform control flow, inside one subgroup.
 * src/gpu/test_gemm_qkv.py and src/gpu/test_window_block.py hold it to the bit. */
#ifdef COSINE_TREE_SHUFFLE
float cosine_reciprocal_lanes(float h[8], bool vit) {
    float partial[2];
    for (uint p = 0u; p < 2u; p++)
        partial[p] = vit
            ? hadd(hfma(h[p], h[p], hmul(h[4u + p], h[4u + p])),
                   hfma(h[2u + p], h[2u + p], hmul(h[6u + p], h[6u + p])))
            : hadd(hfma(h[2u + p], h[2u + p], hmul(h[p], h[p])),
                   hfma(h[6u + p], h[6u + p], hmul(h[4u + p], h[4u + p])));
    float pairs[2], whole[2];
    for (uint p = 0u; p < 2u; p++) pairs[p] = hadd(partial[p], subgroupShuffleXor(partial[p], 2u));
    for (uint p = 0u; p < 2u; p++) whole[p] = hadd(pairs[p], subgroupShuffleXor(pairs[p], 1u));
    float norm = max(hadd(whole[0], whole[1]), half_round(COSINE_NORM_FLOOR));
    return vit ? half_round(1.0 / sqrt(norm)) : half_round(inversesqrt(norm));
}
#endif
