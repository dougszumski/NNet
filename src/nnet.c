/*
 *   nnet.c
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

#include "nnet.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <assert.h>

err_t
network_allocate (network_t * const net)
{
    err_t err = GSL_SUCCESS;

    err |= matrix_array_allocate (&net->nabla_w, &net->nodes);
    err |= matrix_array_allocate (&net->weights, &net->nodes);

    // These arrays are not required for the input layer
    uint32_array_t dimensions = {
            .size = net->nodes.size - 1,
            .data = net->nodes.data + 1
    };
    err |= vector_array_allocate (&net->zs, &dimensions, 0);
    err |= vector_array_allocate (&net->nabla_b, &dimensions, 0);
    err |= vector_array_allocate (&net->output_delta, &dimensions, 0);
    err |= vector_array_allocate (&net->biases, &dimensions, 0);

    // To simplify calculations store a pointer to input vector at -1
    // in the outputs array.
    err |= vector_array_allocate (&net->outputs, &dimensions, 1);

    return err;
}

void
network_free (network_t * const net)
{
    vector_array_free (&net->outputs);
    vector_array_free (&net->zs);
    vector_array_free (&net->nabla_b);
    vector_array_free (&net->output_delta);
    vector_array_free (&net->biases);

    matrix_array_free (&net->nabla_w);
    matrix_array_free (&net->weights);
}

void
network_random_init (network_t * const net, const double var)
{
    gsl_rng * rng = gsl_rng_alloc (gsl_rng_mt19937);

    vector_array_set_rand (&net->biases, rng, var);
    matrix_array_set_rand (&net->weights, rng, var);

    gsl_rng_free (rng);
}

/*
 * Calculate the output vector from the input
 *
 */
void
network_feed_forward (const network_t * const net, const uint8_t store_z)
{
    uint32_t whole_layers = net->nodes.size - 1;

    for (int32_t i = 0; i < whole_layers; ++i)
    {
        // a^l = w^l * a^(l-1) + b^l
        gsl_blas_dgemv (CblasNoTrans, 1.0, net->weights.data[i],
                        net->outputs.data[i - 1], 0.0, net->outputs.data[i]);

        gsl_blas_daxpy (1.0, net->biases.data[i], net->outputs.data[i]);

        if (store_z)
            gsl_vector_memcpy (net->zs.data[i], net->outputs.data[i]);

        vector_vectorise (net->outputs.data[i], &sigmoid);
    }
}

/*
 * 	Stochastic Gradient Descent
 */
void
network_sgd (network_t * const net,
             const data_t * const data,
             const data_t * const test_data)
{
    // Use default random seed of 0
    gsl_rng * rng = gsl_rng_alloc (gsl_rng_mt19937);

    // Index array used to address labels and images in random order
    uint32_t rand_index[data->items];
    for (uint32_t i = 0; i < data->items; ++i) {
        rand_index[i] = i;
    }

    for (uint32_t i = 0; i < net->epochs; ++i)
    {
        // Randomise the index array
        gsl_ran_shuffle (rng, rand_index,
                         sizeof(rand_index) / sizeof(rand_index[0]),
                         sizeof(rand_index[0]));

        network_process_mini_batches (net, data, &rand_index[0],
                                      &network_update_mini_batch);

        uint32_t correct_answers = 0;
        network_evaluate_test_data (net, test_data, &correct_answers);

        printf ("Epoch %i complete, %i/%i correct.\n", i, correct_answers,
                test_data->items);
    }
    gsl_rng_free (rng);
}

void
network_process_mini_batches (network_t * const net,
                              const data_t * const data,
                              const uint32_t * const rand_index,
                              update_batch_f update_batch)
{
    assert(data->items != 0);
    assert(net->mini_batch_size != 0);

    // A slice of randomised indexes for the input data
    uint32_array_t slice = {
            .data = (uint32_t *) rand_index,
            .size = net->mini_batch_size
    };

    uint32_t batches = data->items / net->mini_batch_size;
    printf ("Iterating over %i batches...\n", batches);

    for (uint32_t i = 0; i < batches; ++i) {
        update_batch (net, data, &slice);
        slice.data += slice.size;
    }

    uint32_t remainder = data->items % net->mini_batch_size;
    if (remainder) {
        slice.size = remainder;
        update_batch (net, data, &slice);
    }
}

void
network_update_mini_batch (network_t * const net,
                           const data_t * const data,
                           const uint32_array_t * const slice)
{
    assert(slice->size != 0);

    // Reset batch averages
    vector_array_zero (&net->nabla_b);
    matrix_array_set_zero (&net->nabla_w);

    // Apply SGD to the mini-batch
    for (uint32_t i = 0; i < slice->size; ++i) {
        uint32_t random_index = slice->data[i];
        net->outputs.data[INPUT_INDEX] = data->images.images[random_index];
        network_backpropagate_error (net, data->labels.labels[random_index]);
    }

    // Update weights and biases
    double scale_fac = net->eta / slice->size;

    uint32_t whole_layers = net->nodes.size - 1;
    for (uint32_t i = 0; i < whole_layers; ++i) {
        gsl_matrix_scale (net->nabla_w.data[i], scale_fac);
        gsl_matrix_sub (net->weights.data[i], net->nabla_w.data[i]);

        gsl_vector_scale (net->nabla_b.data[i], scale_fac);
        gsl_vector_sub (net->biases.data[i], net->nabla_b.data[i]);
    }
}

void
network_get_output_error (network_t * const net, const uint8_t label)
{
    uint32_t output_layer_index = net->outputs.size - 1;

    gsl_vector * cost_deriv = gsl_vector_alloc (
            net->outputs.data[output_layer_index]->size);

    cost_derivative (net->outputs.data[output_layer_index], label, cost_deriv);

    gsl_vector_memcpy (net->output_delta.data[output_layer_index],
                       net->zs.data[output_layer_index]);

    vector_vectorise (net->output_delta.data[output_layer_index],
                      &sigmoid_prime);

    gsl_vector_mul (net->output_delta.data[output_layer_index], cost_deriv);

    gsl_vector_free (cost_deriv);
}

/*
 * Accumulate cost function gradients
 */
void
network_accumulate_cfgs (network_t * const net, const int32_t layer)
{
    // Y = alphaX + Y
    gsl_blas_daxpy (1.0, net->output_delta.data[layer],
                    net->nabla_b.data[layer]);

    // A = [delta] * [activations]^T + A
    gsl_blas_dger (1.0, net->output_delta.data[layer],
                   net->outputs.data[layer - 1], net->nabla_w.data[layer]);
}

void
network_backpropagate_error (network_t * const net, const uint8_t label)
{
    network_feed_forward (net, 1);
    network_get_output_error (net, label);

    uint32_t output_layer_index = net->outputs.size - 1;
    network_accumulate_cfgs (net, output_layer_index);

    // Back propagate
    for (int32_t l = output_layer_index - 1; l >= 0; --l)
    {
        gsl_vector_memcpy (net->output_delta.data[l], net->zs.data[l]);

        vector_vectorise (net->output_delta.data[l], &sigmoid_prime);

        gsl_vector * tmp = gsl_vector_alloc (net->output_delta.data[l]->size);

        // Y = alpha(A^T) + beta(Y)
        gsl_blas_dgemv (CblasTrans, 1.0, net->weights.data[l + 1],
                        net->output_delta.data[l + 1], 0.0, tmp);

        // Back-propagated delta
        gsl_vector_mul (net->output_delta.data[l], tmp);

        network_accumulate_cfgs (net, l);

        gsl_vector_free (tmp);
    }
}

void
network_get_output (network_t * const net, uint32_t * const output)
{
    network_feed_forward (net, 0);

    // Returns the lowest index if more than 1.
    *output = gsl_vector_max_index (
            net->outputs.data[net->outputs.size - 1]);
}

void
network_evaluate_test_data (network_t * const net,
                            const data_t * const test_data,
                            uint32_t * const correct_answers)
{
    *correct_answers = 0;
    uint32_t output;

    for (uint32_t i = 0; i < test_data->items; ++i) {
        net->outputs.data[INPUT_INDEX] = test_data->images.images[i];

        network_get_output (net, &output);
        if (output == test_data->labels.labels[i])
            (*correct_answers)++;
    }
}

double
sigmoid (double z)
{
    return 1.0 / (1.0 + exp (-z));
}

double
sigmoid_prime (double z)
{
    double sig_z = sigmoid (z);
    return sig_z * (1.0 - sig_z);
}

void
cost_derivative (const gsl_vector * const output_activations,
                 const uint32_t y,
                 gsl_vector * const cost_derivative)
{
    gsl_vector_memcpy (cost_derivative, output_activations);

    // Subtract expected output (unit vector) from the output
    gsl_vector_set (cost_derivative, y,
                    gsl_vector_get (cost_derivative, y) - 1.0);
}
