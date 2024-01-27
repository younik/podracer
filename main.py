import os
from podracer import launch

if __name__ == '__main__':
    # Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
    # Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
    os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
    os.environ["TF_CUDNN DETERMINISTIC"] = "1"

    launch.main()