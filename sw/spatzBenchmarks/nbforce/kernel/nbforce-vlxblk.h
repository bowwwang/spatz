// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef NBFORCE_VLXBLK_H
#define NBFORCE_VLXBLK_H

#include <stdint.h>

// GROMACS-style 4x4 cluster-pair non-bonded force, VLXBLK arm: per chunk
// of 8 j-clusters one u16 cluster-id vector drives four 128-B field-block
// gathers (x, y, z, q; blk_len = 4 fp32, e32 m1), j-data reused by all 4
// i-atoms; forces accumulate lane-wise in vector registers, reduced and
// read back once per i-cluster.
// fo[c*12 + 3*a + d] = force component d of atom a of i-cluster c.
// list must be a multiple of 16; pairlist carries >= 64 ids of padding.
// cut2_bits: fp32 bit pattern of the squared cutoff.
void nbforce_vlxblk(float *fo, const float *nb_x, const float *nb_y,
                    const float *nb_z, const float *nb_q,
                    const uint16_t *pairlist, const unsigned int nc,
                    const unsigned int list, const uint32_t cut2_bits);

#endif
