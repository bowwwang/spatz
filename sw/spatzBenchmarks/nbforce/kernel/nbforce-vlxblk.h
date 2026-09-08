// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef NBFORCE_VLXBLK_H
#define NBFORCE_VLXBLK_H

#include <stdint.h>

// GROMACS-style 4x4 cluster-pair non-bonded force, VLXBLK arm: one u16
// cluster-id vector per chunk drives four 16-B field-block gathers
// (x, y, z, q; blk_len = 4 fp32), j-data reused by all 4 i-atoms.
// fo[c*12 + 3*a + d] = force component d of atom a of i-cluster c.
void nbforce_vlxblk(float *fo, const float *nb_x, const float *nb_y,
                    const float *nb_z, const float *nb_q,
                    const uint16_t *pairlist, const unsigned int nc,
                    const unsigned int list, const float cut2);

#endif
