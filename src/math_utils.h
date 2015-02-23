/*
 *   math_utils.h
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

#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "errors.h"

#include <stdint.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

typedef struct
{
    uint32_t size;
    uint32_t * data;
} uint32_array_t;

typedef struct
{
    uint32_t size;
    uint32_t offset;
    gsl_vector ** data;
} vector_array_t;

typedef struct
{
    uint32_t size;
    gsl_matrix ** data;
} matrix_array_t;

// For vectorising array
typedef double (*v_func_t) (double);

err_t
vector_array_allocate (vector_array_t * const array,
                       const uint32_array_t * const dimensions,
                       const uint32_t offset);

void
vector_array_zero (vector_array_t * const vector_array);

void
matrix_array_set_zero (matrix_array_t * const matrix_array);

void
vector_array_free (vector_array_t * const vector_array);

err_t
matrix_array_allocate (matrix_array_t * const matrix_array,
                       const uint32_array_t * const structure);

void
matrix_array_free (matrix_array_t * const matrix_array);

void
vector_set_rand (gsl_vector * const vec, const gsl_rng * const rng, double var);

void
vector_array_set_rand (vector_array_t * const vector_array,
                       const gsl_rng * const rng,
                       double var);

void
matrix_set_rand (gsl_matrix * const mat, const gsl_rng * const rng, double var);

void
matrix_array_set_rand (matrix_array_t * const matrix_array,
                       const gsl_rng * const rng,
                       double var);

void
vector_vectorise (gsl_vector * const vec, v_func_t func);

#ifdef __cplusplus
}
#endif

#endif /* MATH_UTILS_H_ */
