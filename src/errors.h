/*
 *   errors.h
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

#ifndef COMMON_H_
#define ERRORS_H_

#include <stdint.h>
#include <gsl/gsl_errno.h>

typedef int32_t err_t;

#define EXIT_MAIN_ON_ERR(x) \
        do { \
            switch(x) { \
                case(GSL_SUCCESS): \
                    break; \
                case(GSL_ENOMEM): \
                    printf("Exiting: Malloc failed.\n"); \
                    return EXIT_FAILURE;  \
                case(GSL_EFAILED): \
                    printf("Exiting: Generic failure, check file paths.\n"); \
                    return EXIT_FAILURE;  \
                default: \
                    printf("Exiting with error code: %i\n", err); \
                    return EXIT_FAILURE; \
            } \
        } while(0)

#define RETURN_ON_ERR(x) \
        do { \
            if(x) { \
                return x; \
            } \
        } while(0)

#define RETURN_ERR_ON_BAD_ALLOC(x) \
        do { \
            if(x == NULL) { \
                return GSL_ENOMEM; \
            } \
        } while(0)

#define RETURN_ERR_ON_NO_FILE(x) \
        do { \
            if(x == NULL) { \
                return GSL_EFAILED; \
            } \
        } while(0)

#endif /* ERRORS_H_ */
