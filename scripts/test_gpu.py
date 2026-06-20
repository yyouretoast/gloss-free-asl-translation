import torch

def main():
    print("Testing PyTorch installation...")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        # Test tensor on GPU
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = x @ y
        print("Successfully ran a tensor multiplication on GPU!")
        print(f"Result device: {z.device}")
    else:
        print("CUDA is NOT available. Running on CPU instead.")
        x = torch.rand(3, 3)
        y = torch.rand(3, 3)
        z = x @ y
        print("Successfully ran a tensor multiplication on CPU.")

if __name__ == "__main__":
    main()
