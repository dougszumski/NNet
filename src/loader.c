/*
 *   loader.c
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

#include "loader.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

err_t
extract_header_line (const uint8_t * const buf)
{
    // Reverse int32 byte order for Intel machines
    // See http://yann.lecun.com/exdb/mnist/
    return (buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
}

err_t
read_all_data (data_t * const data,
               const char * images_file,
               const char * labels_file)
{
    err_t err;

    // Load the training and test data
    err = images_read_data (&data->images, images_file);
    images_print_stats (&data->images);
    RETURN_ON_ERR(err);

    err = labels_read_data (&data->labels, labels_file);
    RETURN_ON_ERR(err);

    labels_print_stats (&data->labels);

    // Keep the individual image and label count for now...
    assert(data->images.num_images == data->labels.num_labels);

    return GSL_SUCCESS;
}

err_t
images_read_data (images_t * const image_data, const char * images_file)
{
    FILE *fp = fopen (images_file, "rb");
    RETURN_ERR_ON_NO_FILE(fp);

    uint8_t buf[IMAGES_HEADER_SIZE_BYTES];
    fread (&buf, 1, IMAGES_HEADER_SIZE_BYTES, fp);

    image_data->magic_num = extract_header_line (buf);
    image_data->num_images = extract_header_line (&buf[4]);
    image_data->rows = extract_header_line (&buf[8]);
    image_data->cols = extract_header_line (&buf[12]);

    uint32_t pixels = image_data->rows * image_data->cols;
    printf ("Pixels per image: %d \n", pixels);

    err_t err;
    err = images_allocate (image_data, pixels);
    RETURN_ON_ERR(err);

    images_load_pixels (image_data, pixels, fp);

    fclose (fp);

    return GSL_SUCCESS;
}

err_t
images_allocate (images_t * const image_data, uint32_t pixels)
{
    image_data->images =
            malloc (image_data->num_images * sizeof(gsl_vector *));
    RETURN_ERR_ON_BAD_ALLOC(image_data->images);

    for (int i = 0; i < image_data->num_images; ++i) {
        image_data->images[i] = gsl_vector_alloc (pixels);
    }

    return GSL_SUCCESS;
}

void
images_free (images_t * const image_data)
{
    for (int i = 0; i < image_data->num_images; ++i) {
        gsl_vector_free (image_data->images[i]);
    }

    free (image_data->images);
}

/*
 * Write image pixels to each vector in the image array.
 *
 * TODO: Set using blocks?
 */
void
images_load_pixels (images_t * const image_data,
                    const uint32_t pixels,
                    FILE * const fp)
{
    for (int i = 0; i < image_data->num_images; ++i) {
        uint8_t buf[pixels];
        fread (&buf, 1, pixels, fp);
        for (int j = 0; j < pixels; ++j) {
            // Normalise the greyscale value to prevent saturation
            // of the sigmoid function
            double tmp = buf[j] / 255.0;
            gsl_vector_set (image_data->images[i], j, tmp);
        }
    }
}

void
images_print_stats (const images_t * const img_data)
{
    printf ("*** Images: ***\n");
    printf ("Magic Num: %d \n", img_data->magic_num);
    printf ("Images   : %d \n", img_data->num_images);
    printf ("Rows     : %d \n", img_data->rows);
    printf ("Columns  : %d \n\n", img_data->cols);
}

err_t
labels_read_data (labels_t * const label_data, const char * labels_file)
{
    FILE *fp = fopen (labels_file, "rb");
    RETURN_ERR_ON_NO_FILE(fp);

    uint8_t buf[LABELS_HEADER_SIZE_BYTES];
    fread (&buf, 1, LABELS_HEADER_SIZE_BYTES, fp);

    label_data->magic_num = extract_header_line (buf);
    label_data->num_labels = extract_header_line (&buf[4]);

    err_t err;
    err = labels_allocate (label_data);
    RETURN_ON_ERR(err);

    uint8_t digit;
    for (uint32_t i = 0; i < label_data->num_labels; ++i) {
        fread (&digit, 1, 1, fp);
        label_data->labels[i] = digit;
    }

    fclose (fp);

    return GSL_SUCCESS;
}

err_t
labels_allocate (labels_t * const label_data)
{
    label_data->labels = malloc (
            label_data->num_labels * sizeof(*(label_data->labels)));
    RETURN_ERR_ON_BAD_ALLOC(label_data->labels);

    return GSL_SUCCESS;
}

void
labels_free (labels_t * const label_data)
{
    free (label_data->labels);
}

void
labels_print_stats (const labels_t * const label_data)
{
    printf ("*** Labels: ***\n");
    printf ("Magic Num: %d \n", label_data->magic_num);
    printf ("Labels  : %d \n\n", label_data->num_labels);
}

/*
 * Split off a chunk from the data to use as test data
 */
void
partition_data (data_t * const data,
                data_t * const test_data,
                const uint32_t chunk_size)
{
    assert(data->images.num_images > chunk_size);

    data->items = data->images.num_images - chunk_size;

    test_data->items = chunk_size;
    test_data->images.images = data->images.images + data->items;
    test_data->labels.labels = data->labels.labels + data->items;
}
