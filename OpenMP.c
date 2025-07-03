#include <omp.h>
#include <stdio.h>
#include <limits.h>

int main() {
    // Define the three datasets as a 2D array
    int datasets[3][10] = {
        {123, 98, 145, 167, 112, 134, 156, 178, 120, 110},
        {140, 132, 115, 109, 150, 148, 160, 172, 125, 138},
        {102, 95, 110, 108, 97, 120, 130, 150, 140, 115}
    };
    int n = 10;

    // Loop through each dataset
    for (int d = 0; d < 3; d++) {
        int* current_dataset = datasets[d];
        
        // Initialize min and max with the first value of the current dataset
        int min_val = current_dataset[0];
        int max_val = current_dataset[0];

        // Start timer
        double start_time = omp_get_wtime();

        // Use OpenMP to find the min and max in parallel for the current dataset
        // The 'reduction' clause handles combining the results from each thread safely.
        // here, we seperate it to multiple threads that will process different parts of the dataset
        // and then combine the results to find the overall min and max.
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
        for (int i = 0; i < n; i++) {
            if (current_dataset[i] < min_val) {
                min_val = current_dataset[i];
            }
            if (current_dataset[i] > max_val) {
                max_val = current_dataset[i];
            }
        }

        // Stop timer
        double end_time = omp_get_wtime();

        // Print the results for the current dataset
        printf("\n--- OpenMP Results for Dataset %d ---\n", d + 1);
        printf("Dataset: ");
        for(int i=0; i<n; i++) {
            printf("%d ", current_dataset[i]);
        }
        printf("\n");
        printf("Highest value (OpenMP): %d\n", max_val);
        printf("Lowest value (OpenMP): %d\n", min_val);
        printf("Execution Time (OpenMP): %f seconds\n", end_time - start_time);
    }

    return 0;
}