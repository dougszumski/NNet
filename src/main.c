/*
 *   main.c
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

#include "error.h"
#include "loader.h"
#include "nnet.h"

#include <stdio.h>

#define VALIDATION_DATA_CHUNK_SIZE 10000
#define MINI_BATCH_SIZE 10
#define EPOCHS 10
#define ETA 3.0
#define RANDOM_VARIANCE 1.0

// Number of nodes in each layer of the network
uint32_t nodes[] = { 784, 30, 10 };

const char * images_file = "./dat/train-images.idx3-ubyte";
const char * labels_file = "./dat/train-labels.idx1-ubyte";

int
main (int argc, const char* argv[])
{
    err_t err;

    printf ("Loading images and labels...\n");
    data_t data;
    err = read_all_data (&data, images_file, labels_file);
    EXIT_MAIN_ON_ERR(err);

    printf ("Setting up network...\n");
    network_t network;
    uint32_t layers = sizeof(nodes) / sizeof(nodes[0]);

    printf ("Node structure: ");
    for (uint32_t i = 0; i < layers - 1; ++i) {
        printf ("%i x ", nodes[i]);
    }
    printf ("%i.\n", nodes[layers - 1]);

    network.nodes.size = layers;
    network.nodes.data = nodes;
    network.epochs = EPOCHS;
    network.mini_batch_size = MINI_BATCH_SIZE;
    network.eta = ETA;

    err = network_allocate (&network);
    EXIT_MAIN_ON_ERR(err);

    printf ("Initialising network...\n");
    network_random_init (&network, RANDOM_VARIANCE);

    // Split off a chunk of data for testing
    data_t test_data;
    partition_data(&data, &test_data, VALIDATION_DATA_CHUNK_SIZE);

    printf ("Stochastic gradient descent...\n");
    network_sgd (&network, &data, &test_data);

    network_free (&network);
    images_free (&data.images);
    labels_free (&data.labels);

    return EXIT_SUCCESS;
}
