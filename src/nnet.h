/*
 *   nnet.h
 *
 *   Copyright 2015 Doug Szumski <d.s.szumski@gmail.com>
 *
 *   This file is part of NNet.
 *
 *   NNet is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   NNet is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with NNet.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NNET_H_
#define NNET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "errors.h"
#include "loader.h"
#include "math_utils.h"

#include <math.h>
#include <stdint.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>

#define INPUT_INDEX -1

typedef struct
{
    double eta;
    uint32_t epochs;
    uint32_t mini_batch_size;
    uint32_array_t nodes;
    vector_array_t outputs; // Input is at [-1]
    vector_array_t zs;
    vector_array_t nabla_b;
    vector_array_t output_delta;
    vector_array_t biases;
    matrix_array_t weights;
    matrix_array_t nabla_w;
} network_t;

typedef void
(*update_batch_f) (network_t * const,
                   const data_t * const,
                   const uint32_array_t * const);

void
network_free (network_t * const network);

err_t
network_allocate (network_t * const network);

void
network_random_init (network_t * const network, const double var);

void
network_sgd (network_t * const network,
             const data_t * const data,
             const data_t * const test_data);

void
network_process_mini_batches (network_t * const network,
                              const data_t * const data,
                              const uint32_t * const rnd_idx,
                              update_batch_f update_batch_f);

void
network_update_mini_batch (network_t * const network,
                           const data_t * const data,
                           const uint32_array_t * const array_slice);

void
network_get_output_error (network_t * const network, const uint8_t label);

void
network_accumulate_cfgs (network_t * const network, const int32_t layer);

void
network_backpropagate_error (network_t * const network, const uint8_t label);

void
network_evaluate_test_data (network_t * const network,
                            const data_t * const test_data,
                            uint32_t * const correct_answers);

void
network_evaluate_output (network_t * const network, uint32_t * const output);

void
network_feed_forward (const network_t * const network, const uint8_t store_z);

double
sigmoid (double z);

double
sigmoid_prime (double z);

void
cost_derivative (const gsl_vector * const output_activations,
                 const uint32_t y,
                 gsl_vector * const cost_derivative);

#ifdef __cplusplus
}
#endif

#endif /* NNET_H_ */
