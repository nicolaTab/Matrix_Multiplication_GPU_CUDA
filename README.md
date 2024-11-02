# Matrix Multiplication with CUDA on GPU

This project demonstrates GPU-accelerated matrix multiplication using CUDA in Python. The code uses shared memory and tiling optimizations to improve performance, particularly beneficial for large matrices. Implemented in a Jupyter notebook, it’s designed to be run on Google Colab, which offers free access to GPU resources.

## Overview

- **Language**: Python
- **Libraries**: Numba, CUDA, NumPy
- **Techniques Used**: Shared memory, Tiling, GPU acceleration
- **Platform**: Google Colab (or a local setup with CUDA-compatible GPU)

## Project Structure

- **Notebook**: `Matrix_Multiplication_GPU_CUDA_Python.ipynb` - The main Jupyter notebook with the code and explanations for matrix multiplication using CUDA.
- **README.md**: A comprehensive guide to understanding, setting up, and running the project.

## Key Features

1. **Baseline Matrix Multiplication**: Basic implementation without shared memory, serving as a reference for performance comparison.
2. **Optimized Matrix Multiplication with Shared Memory**: Uses GPU shared memory for optimized data access, reducing memory access time.
3. **Tiled Matrix Multiplication**: Divides matrices into smaller blocks (tiles) to further optimize memory usage and computation time.
4. **Validation Process**: Compares GPU results with NumPy’s CPU-based results to ensure accuracy.

## Detailed Setup and Requirements

### Prerequisites

- **Google Colab**: The project is designed to be run on Google Colab to utilize GPU resources.
- **Numba and CUDA**: Uses Numba with CUDA for GPU acceleration. Google Colab has these installed, so there’s no need for additional setup.

### Setting Up the Project

#### Option 1: Running on Google Colab
1. **Upload the Notebook to Colab**:
   - Save the `Matrix_Multiplication_GPU_CUDA_Python.ipynb` file to your Google Drive.
   - Open it in Google Colab by navigating to **File > Open notebook > Google Drive**.

2. **Select GPU as Hardware Accelerator**:
   - Go to **Runtime > Change runtime type**.
   - Set **Hardware accelerator** to **GPU**.
   - **Tip**: Always check this step! Forgetting it will make the notebook run on the CPU instead, which will be significantly slower.

#### Option 2: Running Locally
If you have a CUDA-compatible GPU on your local machine:
1. **Install Numba and CUDA**:
   ```bash
   pip install numba
   ```
   Ensure CUDA is installed and the environment variable `NUMBA_CUDA_DRIVER` is set.

2. **Run the notebook** in a Jupyter environment with the GPU enabled.

## Running the Notebook

### Step-by-Step Execution

1. **Initialize the Notebook**: Run the first few cells to import libraries and set matrix size.
2. **Matrix Size Selection**: When prompted, enter the matrix size (e.g., `256`, `512`, `1024`).
   - **Tip**: Larger matrices (e.g., `5000x5000`) will demonstrate the GPU speedup more effectively but may require more time.
3. **Shared Memory Option**: Select `'yes'` or `'no'` when asked if you want to use shared memory.
   - **Recommendation**: Use shared memory for larger matrices to observe faster computation times.
4. **Results**:
   - **Execution Time**: Displayed after each multiplication, showing performance differences.
   - **Validation**: Confirms if the GPU results match NumPy’s results on the CPU.

### Example Usage

For example, with a **256x256** matrix and shared memory enabled:
```plaintext
Enter matrix size (N x N): 256
Use shared memory? (yes/no): yes
Execution time: 0.1234 seconds
Validation successful: GPU and NumPy results are similar.
```

## Performance Insights and Tips

- **Shared Memory**: Enables faster memory access on the GPU, especially useful for large matrices. Expect a noticeable difference with matrices above `1000x1000`.
- **Tiling**: Further optimizes memory usage by splitting the matrix into tiles. This minimizes the number of times data is read from global memory.
- **Execution Time**: Larger matrices or enabling shared memory may increase memory usage, so start with smaller matrices and gradually increase the size to see performance scaling.
  
### Common Issues and Troubleshooting

- **CUDA Error: Driver Not Found**:
  - Make sure the runtime is set to GPU in Colab.
  - If on a local setup, verify that CUDA is correctly installed.

- **Slow Performance**:
  - Check that the GPU is enabled. Running on the CPU will be significantly slower, especially for large matrices.

- **Memory Limits on Colab**:
  - Google Colab may limit GPU memory usage, so avoid setting the matrix size too high. For very large matrices, consider using Colab Pro.

## Validation

The notebook includes a validation step to ensure accuracy:
- **Process**: The GPU results are compared with the CPU results from NumPy.
- **Threshold**: A tolerance of `1e-5` is used to handle minor floating-point differences.
- **Message**: You’ll see "Validation successful: GPU and NumPy results are similar" if results match.

## Sample Results and Expected Performance

Here’s an example of expected performance for different matrix sizes on Google Colab:

| Matrix Size | Shared Memory | Execution Time (approx.) | Validation |
|-------------|---------------|--------------------------|------------|
| 256x256     | No            | ~0.1 sec                 | Success    |
| 256x256     | Yes           | ~0.08 sec                | Success    |
| 5000x5000   | No            | ~2.8 sec                 | Success    |
| 5000x5000   | Yes           | ~2.6 sec                 | Success    |

## License

This project is open-source and licensed under the MIT License. Feel free to fork and modify it for educational or personal use.

---

### Author

Created by Nicola Tabbah. For questions, suggestions, or collaboration, feel free to reach out!

---
