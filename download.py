import kagglehub

# Download latest version
path = kagglehub.dataset_download("mirbektoktogaraev/madrid-real-estate-market")

print("Path to dataset files:", path)