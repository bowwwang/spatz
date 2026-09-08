// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef NBFORCE_RVV_H
#define NBFORCE_RVV_H

#include <stdint.h>

// GROMACS-style 4x4 cluster-pair non-bonded force, plain-RVV baseline:
// canonical element-granular vluxei32 from a PRECOMPUTED expanded
// per-element u32 index array (value = cluster*4 + lane; the kernel
// shifts <<2 to byte offsets), loaded inside the timed region.
// fo[c*12 + 3*a + d] = force component d of atom a of i-cluster c.
void nbforce_rvv(float *fo, const float *nb_x, const float *nb_y,
                 const float *nb_z, const float *nb_q,
                 const uint32_t *pairlist_exp, const unsigned int nc,
                 const unsigned int list, const float cut2);

#endif
