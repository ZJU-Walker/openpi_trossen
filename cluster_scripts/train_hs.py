import torch
import time

def test(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    tensors = []
    # Allocate memory
    for i in range(2):
        tensors.append(torch.randn(5000, 5000, device=device))
    
    # Keep computing
    while True:
        result = torch.matmul(tensors[0], tensors[1])
        time.sleep(0.5)

# Occupy all available GPUs
import threading

num_gpus = torch.cuda.device_count()

threads = []
for gpu_id in range(num_gpus):
    thread = threading.Thread(target=test, args=(gpu_id,))
    thread.start()
    threads.append(thread)

# Wait for all threads
for thread in threads:
    thread.join()