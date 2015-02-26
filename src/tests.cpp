/*
 *   tests.cpp
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

#define CATCH_CONFIG_MAIN

#include <cstdlib>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>

#include "errors.h"
#include "math_utils.h"
#include "catch.hpp"
#include "nnet.h"
#include "loader.h"

#define BIG_NUM 9999.0

TEST_CASE( "Extract header line", "[loader]" )
{
    uint8_t int32_field[4] = { 0x00, 0x00, 0x08, 0x03 };

    int32_t test = extract_header_line (int32_field);

    REQUIRE(test == 2051);
}

TEST_CASE( "Images read data", "[loader]" )
{
    images_t img_data;
    err_t err;

    // Check file IO error is caught
    err = images_read_data (&img_data, "");
    REQUIRE(err > 0);

    // Read a known valid file
    err = images_read_data (&img_data, "./dat/train-images-idx3-ubyte");
    REQUIRE(err == 0);

    // Check the file header
    REQUIRE(img_data.magic_num == 2051);
    REQUIRE(img_data.num_images == 60000);
    REQUIRE(img_data.rows == 28);
    REQUIRE(img_data.cols == 28);

    // Check the image arrays have been allocated
    for (int32_t i = 0; i < img_data.num_images; ++i) {
        REQUIRE(img_data.images[i]->size == 784);
    }

    // Check a few bytes are in the right place in the final image
    gsl_vector * final_image = img_data.images[img_data.num_images - 1];
    REQUIRE(gsl_vector_get (final_image, 656) == Approx (0x2C / 255.0));
    REQUIRE(gsl_vector_get (final_image, 657) == Approx (0x00 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 658) == Approx (0x00 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 659) == Approx (0x00 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 679) == Approx (0x49 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 680) == Approx (0xC1 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 681) == Approx (0xC5 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 682) == Approx (0x86 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 683) == Approx (0x00 / 255.0));
    REQUIRE(gsl_vector_get (final_image, 684) == Approx (0x00 / 255.0));
}

TEST_CASE( "Labels read data", "[loader]" )
{
    labels_t lbl_data;
    err_t err;

    // Check file IO error is caught
    err = labels_read_data (&lbl_data, "");
    REQUIRE(err > 0);

    // Read a known valid file
    err = labels_read_data (&lbl_data, "./dat/train-labels-idx1-ubyte");
    REQUIRE(err == 0);

    // Check the file header
    REQUIRE(lbl_data.magic_num == 2049);
    REQUIRE(lbl_data.num_labels == 60000);

    // Check a few labels
    REQUIRE(lbl_data.labels[0] == 0x05);
    REQUIRE(lbl_data.labels[1] == 0x00);
    REQUIRE(lbl_data.labels[2] == 0x04);
    REQUIRE(lbl_data.labels[3] == 0x01);
    REQUIRE(lbl_data.labels[4] == 0x09);
    // ...
    REQUIRE(lbl_data.labels[lbl_data.num_labels - 5] == 0x08);
    REQUIRE(lbl_data.labels[lbl_data.num_labels - 4] == 0x03);
    REQUIRE(lbl_data.labels[lbl_data.num_labels - 3] == 0x05);
    REQUIRE(lbl_data.labels[lbl_data.num_labels - 2] == 0x06);
    REQUIRE(lbl_data.labels[lbl_data.num_labels - 1] == 0x08);
}

TEST_CASE( "Sigmoid function", "[nnet]" )
{
    REQUIRE(sigmoid (0.0) == Approx (0.5f));
    REQUIRE(sigmoid(BIG_NUM) == Approx(1.0));
    REQUIRE(sigmoid(-BIG_NUM) == Approx(0.0));
}

TEST_CASE( "Sigmoid prime function", "[nnet]" )
{
    REQUIRE(sigmoid_prime (0.0) == Approx (0.25f));
    REQUIRE(sigmoid_prime(BIG_NUM) == Approx(0.0));
    REQUIRE(sigmoid_prime(-BIG_NUM) == Approx(0.0));
}

TEST_CASE( "Vectorise function", "[nnet]" )
{
    gsl_vector * vec = gsl_vector_alloc (3);

    gsl_vector_set (vec, 0, 0.0);
    gsl_vector_set (vec, 1, BIG_NUM);
    gsl_vector_set (vec, 2, -BIG_NUM);

    vector_vectorise (vec, &sigmoid);

    REQUIRE(gsl_vector_get (vec, 0) == Approx (0.5f));
    REQUIRE(gsl_vector_get (vec, 1) == Approx (1.0));
    REQUIRE(gsl_vector_get (vec, 2) == Approx (0.0));

    gsl_vector_free (vec);
}

void
mini_batch_test (network_t * const network,
                 const data_t * const data,
                 const uint32_array_t * const array_slice)
{
    for (uint32_t i = 0; i < array_slice->size; ++i) {
        REQUIRE(*(array_slice->data + i) == i);
    }
}

TEST_CASE( "Iterate over mini batches", "[nnet]" )
{
    // TODO: Assert num images = num labels and store number only once.
    data_t data;
    network_t network;

    // Perfectly divisible
    data.items = 4;
    network.mini_batch_size = 2;
    uint32_t pd[] = { 0, 1, 0, 1 };

    network_process_mini_batches (&network, &data, &pd[0], &mini_batch_test);

    // Divisible, with remainder
    data.items = 8;
    network.mini_batch_size = 3;
    uint32_t dwr[] = { 0, 1, 2, 0, 1, 2, 0, 1 };

    network_process_mini_batches (&network, &data, &dwr[0], &mini_batch_test);

    // Non-divisible, with remainder
    data.items = 5;
    network.mini_batch_size = 6;
    uint32_t ndwr[] =  { 0, 1, 2, 3, 4 };

    network_process_mini_batches (&network, &data, &ndwr[0], &mini_batch_test);
}

TEST_CASE ("Cost derivative", "[nnet]")
{
    const uint32_t output_size = 4;

    gsl_vector * output_activations = gsl_vector_calloc (output_size);
    gsl_vector_set (output_activations, 3, 0.9f);
    gsl_vector_set (output_activations, 1, 0.1f);

    gsl_vector * res = gsl_vector_alloc (output_size);
    uint8_t y = 3;

    cost_derivative (output_activations, y, res);

    REQUIRE(gsl_vector_get (res, 0) == Approx (0.0));
    REQUIRE(gsl_vector_get (res, 1) == Approx (0.1f));
    REQUIRE(gsl_vector_get (res, 2) == Approx (0.0));
    REQUIRE(gsl_vector_get (res, 3) == Approx (-0.1f));

    gsl_vector_free (output_activations);
    gsl_vector_free (res);
}

TEST_CASE ("Get output error", "[nnet]")
{
    network_t network;

    uint32_t nodes[] = { 2, 3, 2 };
    uint32_t layers = sizeof(nodes) / sizeof(nodes[0]);
    network.nodes.data = nodes;
    network.nodes.size = layers;
    network_allocate (&network);

    uint32_t label = 1;
    uint32_t output_index = layers - 2;

    gsl_vector_set (network.outputs.data[output_index], 0, 0.2f);
    gsl_vector_set (network.outputs.data[output_index], 1, 0.9f);

    gsl_vector_set (network.zs.data[output_index], 0, 0.5f);
    gsl_vector_set (network.zs.data[output_index], 1, 0.1f);

    network_get_output_error (&network, label);

    REQUIRE(gsl_vector_get (network.output_delta.data[output_index], 0)
            == Approx (0.047f));
    REQUIRE(gsl_vector_get (network.output_delta.data[output_index], 1)
            == Approx (-0.02494));

    network_free (&network);
}

TEST_CASE ("Accumulate cost function gradients", "[nnet]")
{
	network_t network;
	uint32_t nodes[] = { 2, 3, 2 };
    uint32_t layers = sizeof(nodes) / sizeof(nodes[0]);
    network.nodes.data = nodes;
    network.nodes.size = layers;
	network_allocate(&network);

	uint32_t output_index = layers - 2;

    gsl_vector_set (network.output_delta.data[output_index], 0, 0.5f);
    gsl_vector_set (network.output_delta.data[output_index], 1, 0.1f);

    // Use non-zero values to check accumulation for the average
    gsl_vector_set (network.nabla_b.data[output_index], 0, 1.0);
    gsl_vector_set (network.nabla_b.data[output_index], 1, 2.0);

    gsl_matrix_set (network.nabla_w.data[output_index], 0, 0, 1.0);
    gsl_matrix_set (network.nabla_w.data[output_index], 0, 1, 2.0);
    gsl_matrix_set (network.nabla_w.data[output_index], 0, 2, 3.0);
    gsl_matrix_set (network.nabla_w.data[output_index], 1, 0, 4.0);
    gsl_matrix_set (network.nabla_w.data[output_index], 1, 1, 5.0);
    gsl_matrix_set (network.nabla_w.data[output_index], 1, 2, 6.0);

    gsl_vector_set (network.outputs.data[output_index - 1], 0, 1.0);
    gsl_vector_set (network.outputs.data[output_index - 1], 1, 2.0);
    gsl_vector_set (network.outputs.data[output_index - 1], 2, 3.0);

    network_accumulate_cfgs (&network, output_index);

    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 0, 0)
            == Approx (1.5f));
    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 0, 1)
            == Approx (3.0));
    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 0, 2)
            == Approx (4.5f));
    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 1, 0)
            == Approx (4.1f));
    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 1, 1)
            == Approx (5.2f));
    REQUIRE(gsl_matrix_get (network.nabla_w.data[output_index], 1, 2)
            == Approx (6.3f));

    network_free (&network);
}

TEST_CASE( "Feed forward", "[nnet]" )
{
	/*
	 * Setup a simple network, where the bias is chosen such that
	 * z from each layer is 0.0.
	 *
	 * TODO: Extend to test non-trivial weights?
	 */

    network_t network;
    uint32_t nodes[] = { 2, 3, 2 };
    uint32_t layers = sizeof(nodes) / sizeof(nodes[0]);
    network.nodes.data = nodes;
    network.nodes.size = layers;
    network_allocate (&network);

    uint32_t output_index = layers - 2;

    // Normally this points at an input image
    network.outputs.data[INPUT_INDEX] = gsl_vector_alloc (
            network.nodes.data[0]);

    gsl_vector_set_all (network.outputs.data[INPUT_INDEX], 1.0);

    gsl_matrix_set_all (network.weights.data[output_index - 1], 1.0);
    gsl_matrix_set_all (network.weights.data[output_index], 1.0);

    gsl_vector_set_all (network.biases.data[output_index - 1], -2.0);
    gsl_vector_set_all (network.biases.data[output_index], -1.5f);

    gsl_vector_set_zero (network.outputs.data[output_index - 1]);
    gsl_vector_set_zero (network.outputs.data[output_index]);

    network_feed_forward (&network, 1);

    // Inputs
    REQUIRE(gsl_vector_get (network.outputs.data[INPUT_INDEX], 0)
            == Approx (1.0));
    REQUIRE(gsl_vector_get (network.outputs.data[INPUT_INDEX], 1)
            == Approx (1.0));

    // Middle layer
    REQUIRE(gsl_vector_get (network.zs.data[output_index - 1], 0)
            == Approx (0.0));
    REQUIRE(gsl_vector_get (network.zs.data[output_index - 1], 1)
            == Approx (0.0));
    REQUIRE(gsl_vector_get (network.zs.data[output_index - 1], 2)
            == Approx (0.0));
    REQUIRE(gsl_vector_get (network.outputs.data[output_index - 1], 0)
            == Approx (0.5f));
    REQUIRE(gsl_vector_get (network.outputs.data[output_index - 1], 1)
            == Approx (0.5f));
    REQUIRE(gsl_vector_get (network.outputs.data[output_index - 1], 2)
            == Approx (0.5f));

    // Output
    REQUIRE(gsl_vector_get (network.zs.data[output_index], 0)
            == Approx (0.0));
    REQUIRE(gsl_vector_get (network.zs.data[output_index], 1)
            == Approx (0.0));
    REQUIRE(gsl_vector_get (network.outputs.data[output_index], 0)
            == Approx (0.5f));
    REQUIRE(gsl_vector_get (network.outputs.data[output_index], 1)
            == Approx (0.5f));

    network_free (&network);
}


TEST_CASE("gsl_blas_sger", "[GSL]")
{
	gsl_vector * a = gsl_vector_alloc(2);
	gsl_vector_set(a, 0, 1.0);
	gsl_vector_set(a, 1, 2.0);

	gsl_vector * b = gsl_vector_alloc(3);
	gsl_vector_set(b, 0, 3.0);
	gsl_vector_set(b, 1, 4.0);
	gsl_vector_set(b, 2, 5.0);

	gsl_matrix * c = gsl_matrix_calloc(2, 3);

	gsl_blas_dger(1.0, a, b, c);

	REQUIRE(gsl_matrix_get(c, 0, 0) == Approx(3.0));
	REQUIRE(gsl_matrix_get(c, 0, 1) == Approx(4.0));
	REQUIRE(gsl_matrix_get(c, 0, 2) == Approx(5.0));
	REQUIRE(gsl_matrix_get(c, 1, 0) == Approx(6.0));
	REQUIRE(gsl_matrix_get(c, 1, 1) == Approx(8.0));
	REQUIRE(gsl_matrix_get(c, 1, 2) == Approx(10.0));
}
