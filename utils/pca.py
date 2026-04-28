import pickle
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
def pca_embedding(file,dim=64,out_dir='pca_embeddings'):
    """
    Perform PCA on item embeddings from a file and save the reduced embeddings.
    Args:
        file (str): Path to the input file containing item embeddings.
        dim (int): Number of dimensions for PCA reduction.
        out_dir (str): Directory to save the PCA reduced embeddings.
    """
    item_embeddings = []
    with open(file, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            embedding = list(map(np.float32, parts[2].split(',')))
            item_embeddings.append(embedding)
    pca = PCA(n_components=dim)
    pca_item_emb = pca.fit_transform(item_embeddings)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, f"pca_itm_emb{dim}.pkl"), "wb") as f:
        pickle.dump(pca_item_emb, f)
        
def topk_embedding(file, k=64, out_dir='topk_embeddings'):
    """
    Extract top-k item embeddings from a file and save them.
    Args:
        file (str): Path to the input file containing item embeddings.
        k (int): Number of top embeddings to extract.
        out_dir (str): Directory to save the top-k embeddings.
    """
    item_embeddings = []
    with open(file, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            embedding = list(map(np.float32, parts[2].split(',')))
            item_embeddings.append(embedding)
    top_k_embeddings = item_embeddings[:k]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, f"topk_itm_emb{k}.pkl"), "wb") as f:
        pickle.dump(top_k_embeddings, f)

def main():
    """
     pip install -U scikit-learn
     sh: python pca.py --file path/to/your/item_embeddings.txt --dim 64 --k 100 --out_dir embeddings
    """
    import argparse
    parser = argparse.ArgumentParser(description="PCA and Top-K Embedding Extraction")
    parser.add_argument('--file', type=str, required=True, help='Path to the input file containing item embeddings')
    parser.add_argument('--dim', type=int, default=64, help='Number of dimensions for PCA reduction')
    parser.add_argument('--k', type=int, default=100, help='Number of top embeddings to extract')
    parser.add_argument('--out_dir', type=str, default='embeddings', help='Directory to save the embeddings')
    
    args = parser.parse_args()
    
    pca_embedding(args.file, dim=args.dim, out_dir=args.out_dir)
    topk_embedding(args.file, k=args.k, out_dir=args.out_dir)
    
    # example of how to load the embeddings in a PyTorch model
    # item_emb = pickle.load(open(filename, "rb"))
    # item_emb = np.insert(item_emb, 0, values=np.zeros((1, item_emb.shape[1])), axis=0)
    # item_emb = np.concatenate([item_emb, np.zeros((1, item_emb.shape[1]))], axis=0)
    # self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(item_emb))    
        
if __name__ == "__main__":
    main()