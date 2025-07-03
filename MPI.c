#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL); // Initialize the MPI environment

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process

    // Define the three datasets in a 2D array
    int all_datasets[3][10] = {
        {123, 98, 145, 167, 112, 134, 156, 178, 120, 110},
        {140, 132, 115, 109, 150, 148, 160, 172, 125, 138},
        {102, 95, 110, 108, 97, 120, 130, 150, 140, 115}
    };
    int n = 10;
    int elements_per_proc = n / world_size;
    
    // Buffer for the root process to hold the current dataset
    int *current_dataset = NULL;
    if (world_rank == 0) {
        current_dataset = (int*)malloc(sizeof(int) * n);
    }
    
    // Subarray for each process to receive its chunk
    int *sub_array = (int *)malloc(sizeof(int) * elements_per_proc);

    // Loop through each dataset
    for (int d = 0; d < 3; d++) {
        // The root process loads the current dataset
        if (world_rank == 0) {
            for(int i=0; i<n; i++) {
                current_dataset[i] = all_datasets[d][i];
            }
        }

        double start_time, end_time;
        if (world_rank == 0) {
            start_time = MPI_Wtime();
        }

        // Scatter the current dataset from the root process to all processes
        MPI_Scatter(current_dataset, elements_per_proc, MPI_INT, sub_array,
                    elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

        // Each process computes the min and max of its subarray
        int local_min = sub_array[0];
        int local_max = sub_array[0];
        for (int i = 1; i < elements_per_proc; i++) {
            if (sub_array[i] < local_min) {
                local_min = sub_array[i];
            }
            if (sub_array[i] > local_max) {
                local_max = sub_array[i];
            }
        }

        // Reduce the local values to a global min and max on the root process
        int global_min;
        int global_max;
        MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        // The root process prints the result for the current dataset
        if (world_rank == 0) {
            end_time = MPI_Wtime();
            printf("\n--- MPI Results for Dataset %d ---\n", d + 1);
            printf("Dataset: ");
            for(int i=0; i<n; i++) {
                printf("%d ", current_dataset[i]);
            }
            printf("\n");
            printf("Highest value (MPI): %d\n", global_max);
            printf("Lowest value (MPI): %d\n", global_min);
            printf("Execution Time (MPI): %f seconds\n", end_time - start_time);
        }
    }

    MPI_Finalize();
    
    // Clean up allocated memory
    if (world_rank == 0) {
        free(current_dataset);
    }
    free(sub_array);

    return 0;
}