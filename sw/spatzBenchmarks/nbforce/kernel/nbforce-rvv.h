// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef NBFORCE_RVV_H
#define NBFORCE_RVV_H

#include <stdint.h>

// GROMACS-style 4x4 cluster-pair non-bonded force (fp32), plain-RVV baseline:
// same kernel as the VLXBLK arm except that each chunk of 8 j-clusters is
// fetched with one 128-B load of 32 precomputed per-atom u32 element indices
// (nb_list_exp, shifted to byte offsets) plus four vluxei32 element gathers.
// fo[c*12 + 3*a + d]; list a multiple of 16; nb_list_exp padded by >= 256.
void nbforce_rvv(float *fo, const float *nb_x, const float *nb_y,
                 const float *nb_z, const float *nb_q,
                 const uint32_t *pairlist_exp, const unsigned int nc,
                 const unsigned int list, const float cut2);

#endif
