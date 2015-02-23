/*
 *   math_utils.c
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

#include "math_utils.h"

#include <assert.h>

/*
 * Allocates an array of i pointers to vectors where the dimension of the i'th
 * vector is given by dimensions.data[i], and i by dimensions.size.
 *
 * Additional pointers before the 0th vector can be allocated with a finite
 * offset. This is used for negative indexing of the array.
 */
err_t
vector_array_allocate (vector_array_t * const array,
                       const uint32_array_t * const dimensions,
                       const uint32_t offset)
{
    array->size = dimensions->size;
    array->offset = offset;
    array->data = malloc (sizeof(gsl_vector *) * (array->size + offset));
    RETURN_ERR_ON_BAD_ALLOC(array->data);
    array->data += offset;

    for (uint32_t i = 0; i < array->size; ++i) {
        array->data[i] = gsl_vector_alloc (dimensions->data[i]);
    }

    return GSL_SUCCESS;
}

void
vector_array_zero (vector_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        gsl_vector_set_zero (array->data[i]);
    }
}

void
matrix_array_set_zero (matrix_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        gsl_matrix_set_zero (array->data[i]);
    }
}

void
vector_array_free (vector_array_t * const array)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        gsl_vector_free (array->data[i]);
    }

    free (array->data - array->offset);
}

/*
 * Allocates an array of i pointers to matrices where the dimensions of the
 * i'th matrix are: dimensions.data[i + 1] x dimensions.data[i], and i is
 * defined by dimensions.size.
 */
err_t
matrix_array_allocate (matrix_array_t * const array,
                       const uint32_array_t * const dimensions)
{
    array->size = dimensions->size - 1;
    array->data = malloc (sizeof(gsl_matrix *) * array->size);
    RETURN_ERR_ON_BAD_ALLOC(array->data);

    for (uint32_t i = 0; i < array->size; ++i) {
        array->data[i] = gsl_matrix_alloc (dimensions->data[i + 1],
                                           dimensions->data[i]);
    }

    return GSL_SUCCESS;
}

void
matrix_array_free (matrix_array_t * const matrix)
{
    for (uint32_t i = 0; i < matrix->size; ++i) {
        gsl_matrix_free (matrix->data[i]);
    }

    free (matrix->data);
}

void
vector_set_rand (gsl_vector * const vec, const gsl_rng * const rng, double var)
{
    for (uint32_t i = 0; i < vec->size; ++i) {
        gsl_vector_set (vec, i, gsl_ran_gaussian (rng, var));
    }
}

void
vector_array_set_rand (vector_array_t * const array,
                       const gsl_rng * const rng,
                       double var)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        vector_set_rand (array->data[i], rng, var);
    }
}

void
matrix_set_rand (gsl_matrix * const mat, const gsl_rng * const rng, double var)
{
    for (uint32_t i = 0; i < mat->size1; ++i) {
        for (uint32_t j = 0; j < mat->size2; ++j) {
            gsl_matrix_set (mat, i, j, gsl_ran_gaussian (rng, var));
        }
    }
}

void
matrix_array_set_rand (matrix_array_t * const array,
                       const gsl_rng * const rng,
                       double var)
{
    for (uint32_t i = 0; i < array->size; ++i) {
        matrix_set_rand (array->data[i], rng, var);
    }
}

void
vector_vectorise (gsl_vector * const vec, v_func_t func)
{
    for (int i = 0; i < vec->size; ++i) {
        double tmp = (*func) (gsl_vector_get (vec, i));
        gsl_vector_set (vec, i, tmp);
    }
}
