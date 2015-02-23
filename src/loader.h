/*
 *   loader.h
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

#ifndef LOADER_H_
#define LOADER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "errors.h"

#include <gsl/gsl_matrix.h>

#define IMAGES_HEADER_SIZE_BYTES 16
#define LABELS_HEADER_SIZE_BYTES 8

typedef struct
{
    int32_t magic_num;
    int32_t num_images;
    int32_t rows;
    int32_t cols;
    gsl_vector ** images;
} images_t;

typedef struct
{
    int32_t magic_num;
    int32_t num_labels;
    uint8_t * labels;
} labels_t;

typedef struct
{
    labels_t labels;
    images_t images;
    uint32_t items;
} data_t;

err_t
extract_header_line (const uint8_t * const buf);

err_t
read_all_data (data_t * const data,
               const char * images_file,
               const char * labels_file);

err_t
images_read_data (images_t * const image_data, const char * images_file);

err_t
images_allocate (images_t * const image_data, uint32_t pixels);

void
images_free (images_t * const image_data);

void
images_load_pixels (images_t * const image_data,
                    const uint32_t pixels,
                    FILE * const fp);

void
images_print_stats (const images_t * const image_data);

err_t
labels_read_data (labels_t * const label_data, const char * labels_file);

err_t
labels_allocate (labels_t * const label_data);

void
labels_free (labels_t * const label_data);

void
labels_print_stats (const labels_t * const label_data);

void
partition_data (data_t * const data,
                data_t * const test_data,
                const uint32_t chunk_size);

#ifdef __cplusplus
}
#endif

#endif /* LOADER_H_ */
