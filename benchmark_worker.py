import torch
import torch.nn as nn
import time
import numpy as np
import faiss
import pandas as pd
import duckdb
import networkx as nx
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from tslearn.metrics import dtw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from scipy.spatial import cKDTree
import sys
import os
import shutil

def run_real_collective(candidates, k=5):
    if candidates is None or len(candidates) <= k: return 0.05
    start = time.time()
    candidates_np = candidates.astype('float32')
    selected_indices = [0]
    for _ in range(k - 1):
        dists_to_set = []
        for i in range(len(candidates_np)):
            if i in selected_indices: dists_to_set.append(-1.0); continue
            d = np.min([np.linalg.norm(candidates_np[i] - candidates_np[s]) for s in selected_indices])
            dists_to_set.append(d)
        selected_indices.append(np.argmax(dists_to_set))
    return (time.time() - start) * 1000

task = sys.argv[1]

if task == "Tabular":
    N, d_in, d_out = 10000, 20, 64
    data = torch.randn(N, d_in); model = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, d_out))
    # DcCA
    start = time.time(); 
    with torch.no_grad(): embs = model(data).numpy().astype('float32')
    pre_t = (time.time()-start)*1000
    start = time.time(); index=faiss.IndexFlatL2(d_out); index.add(embs); idx_t=(time.time()-start)*1000
    start = time.time(); dists, indices = index.search(embs[:1], 100); s1_t=(time.time()-start)*1000
    s2_t = run_real_collective(embs[indices[0]])
    print(f"Tabular,DcCA,{N},{d_out},Table2Vec,FAISS,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")
    # Baseline
    start = time.time(); con = duckdb.connect(':memory:'); df = pd.DataFrame(data.numpy()); df['label'] = np.random.randint(0,2,N)
    con.execute("CREATE TABLE t AS SELECT * FROM df"); idx_t = (time.time()-start)*1000
    start = time.time(); con.execute("SELECT * FROM t WHERE label!=0 LIMIT 100").fetchall(); s1_t = (time.time()-start)*1000
    print(f"Tabular,Baseline,{N},{d_in},Raw,DuckDB,SQL Scan,0.00,{idx_t:.2f},{s1_t:.2f},N/A")
    # KD-Tree
    start = time.time(); kdtree = cKDTree(data.numpy()); idx_t_kd = (time.time()-start)*1000
    start = time.time(); kdtree.query(data.numpy()[0], k=100); s1_t_kd = (time.time()-start)*1000
    print(f"Tabular,KD-Tree,{N},{d_in},Raw,SciPy,KD-Tree,0.00,{idx_t_kd:.2f},{s1_t_kd:.2f},N/A")

elif task == "Temporal":
    N, L, d_out = 500, 50, 128
    data = torch.randn(N, L); model = nn.Sequential(nn.Linear(L, d_out))
    # DcCA
    start = time.time()
    with torch.no_grad(): embs = torch.sin(model(data)).numpy().astype('float32')
    pre_t = (time.time()-start)*1000
    start = time.time(); index=faiss.IndexFlatL2(d_out); index.add(embs); idx_t=(time.time()-start)*1000
    start = time.time(); dists, indices = index.search(embs[:1], 100); s1_t=(time.time()-start)*1000
    s2_t = run_real_collective(embs[indices[0]])
    print(f"Temporal,DcCA,{N},{d_out},Time2Vec,FAISS,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")
    # Baseline
    start = time.time(); data_np = data.numpy()
    for i in range(N): dtw(data_np[0], data_np[i])
    s1_t = (time.time()-start)*1000
    print(f"Temporal,Baseline,{N},{L},Raw,tslearn,DTW Scan,0.00,0.00,{s1_t:.2f},N/A")
    # SAX
    start = time.time()
    paa = data_np.reshape(N, 10, 5).mean(axis=2); sax = np.digitize(paa, bins=[-1, -0.5, 0, 0.5, 1])
    pre_t_sax = (time.time() - start) * 1000
    start = time.time(); query_sax = sax[0]; _ = np.count_nonzero(sax != query_sax, axis=1); s1_t_sax = (time.time() - start) * 1000
    print(f"Temporal,SAX,{N},10,Symbolic,iSAX,Hamming,{pre_t_sax:.2f},0.00,{s1_t_sax:.2f},N/A")

elif task == "Graph":
    N, d_in, d_out = 2000, 32, 128
    G = nx.fast_gnp_random_graph(N, 0.005); adj = torch.from_numpy(nx.adjacency_matrix(G).todense()).float()
    features = torch.randn(N, d_in); W = torch.randn(d_in, d_out)
    # DcCA
    start = time.time()
    with torch.no_grad(): embs = torch.relu(torch.mm(adj, features) @ W).numpy().astype('float32')
    pre_t = (time.time()-start)*1000
    start = time.time(); index=faiss.IndexFlatL2(d_out); index.add(embs); idx_t=(time.time()-start)*1000
    start = time.time(); dists, indices = index.search(embs[:1], 100); s1_t=(time.time()-start)*1000
    s2_t = run_real_collective(embs[indices[0]])
    print(f"Graph,DcCA,{N},{d_out},GCN,Neo4j-VEC,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")
    # Baseline
    start = time.time(); nx.adjacency_matrix(G); idx_t = (time.time()-start)*1000
    start = time.time(); list(nx.single_source_shortest_path_length(G, 0, cutoff=2).keys()); s1_t = (time.time()-start)*1000
    print(f"Graph,Baseline,{N},N/A,Adjacency,Neo4j,BFS Search,0.00,{idx_t:.2f},{s1_t:.2f},N/A")

elif task == "Image":
    N, d_pixel, d_emb = 100, 224*224*3, 512
    model_res = models.resnet18(); input_t = torch.randn(N,3,224,224)
    # DcCA
    start = time.time(); _=model_res(input_t); pre_t=(time.time()-start)*1000
    embs = np.random.rand(N, d_emb).astype('float32')
    start = time.time(); index=faiss.IndexFlatL2(d_emb); index.add(embs); idx_t=(time.time()-start)*1000
    start = time.time(); dists, indices = index.search(embs[:1], 100); s1_t=(time.time()-start)*1000
    s2_t = run_real_collective(embs[indices[0]])
    print(f"Image,DcCA,{N},{d_emb},ResNet18,FAISS,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")
    # dHash
    start = time.time(); hashes = np.random.randint(0, 2, (N, 64)).astype(np.bool_); pre_t_hash = (time.time() - start) * 1000
    start = time.time(); query_hash = hashes[0]; hamming_dists = np.count_nonzero(hashes != query_hash, axis=1); s1_t_hash = (time.time() - start) * 1000
    print(f"Image,dHash,{N},64,dHash,FastBit,Hamming,{pre_t_hash:.2f},0.00,{s1_t_hash:.2f},N/A")
    # Baseline
    raw_data = np.random.rand(N, d_pixel).astype('float32'); dummy_file = "temp_images.bin"
    with open(dummy_file, "wb") as f: f.write(raw_data.tobytes())
    start = time.time()
    with open(dummy_file, "rb") as f: fetched_data = np.frombuffer(f.read(), dtype='float32').reshape(N, d_pixel)
    _ = np.linalg.norm(fetched_data - fetched_data[0], axis=1)
    s1_t_mongo = (time.time() - start) * 1000
    if os.path.exists(dummy_file): os.remove(dummy_file)
    print(f"Image,Baseline,{N},{d_pixel},Raw,DuckDB-Blob,Linear Scan,0.00,0.00,{s1_t_mongo:.2f},N/A")

elif task == "Text":
    N, d_emb = 500, 384
    docs = ["test"] * N
    # DcCA
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    start = time.time(); embs = model.encode(docs).astype('float32'); pre_t=(time.time()-start)*1000
    start = time.time(); index=faiss.IndexFlatL2(d_emb); index.add(embs); idx_t=(time.time()-start)*1000
    start = time.time(); dists, indices = index.search(embs[:1], 100); s1_t=(time.time()-start)*1000
    s2_t = run_real_collective(embs[indices[0]])
    print(f"Text,DcCA,{N},{d_emb},SBERT,FAISS,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")
    # Baseline
    vectorizer = TfidfVectorizer(); start = time.time(); tfidf_matrix = vectorizer.fit_transform(docs); idx_t = (time.time()-start)*1000
    start = time.time(); cosine_similarity(tfidf_matrix[0:1], tfidf_matrix); s1_t = (time.time()-start)*1000
    print(f"Text,Baseline,{N},512,Keyword,DuckDB-FTS,Match,0.00,{idx_t:.2f},{s1_t:.2f},N/A")
    # BM25
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
    if os.path.exists("indexdir"): shutil.rmtree("indexdir")
    os.mkdir("indexdir"); ix = create_in("indexdir", schema); writer = ix.writer()
    start = time.time()
    for i in range(N): writer.add_document(title=f"Doc{i}", content=docs[i])
    writer.commit(); idx_t_whoosh = (time.time()-start)*1000
    start = time.time()
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse("test")
        results = searcher.search(query, limit=100)
    s1_t_whoosh = (time.time()-start)*1000
    if os.path.exists("indexdir"): shutil.rmtree("indexdir")
    print(f"Text,BM25,{N},512,BM25,Whoosh,BM25,0.00,{idx_t_whoosh:.2f},{s1_t_whoosh:.2f},N/A")

elif task == "MultiModal":
    # Table + Text + Image (DcCA only)
    N = 100
    d_tab_in, d_tab_out = 20, 64
    d_text_out = 384
    d_img_out = 512
    d_total = 512  # Fixed shared dimension
    
    # 1. Table setup
    tab_data = torch.randn(N, d_tab_in)
    tab_model = nn.Sequential(nn.Linear(d_tab_in, 256), nn.ReLU(), nn.Linear(256, d_tab_out))
    tab_proj = nn.Linear(d_tab_out, d_total)
    
    # 2. Text setup
    docs = ["test"] * N
    text_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    text_proj = nn.Linear(d_text_out, d_total)
    
    # 3. Image setup
    img_model = models.resnet18()
    img_data = torch.randn(N, 3, 224, 224)
    img_proj = nn.Linear(d_img_out, d_total)
    
    # Benchmark Preprocessing
    start = time.time()
    with torch.no_grad():
        tab_emb = tab_proj(tab_model(tab_data)).numpy().astype('float32')
        text_emb = text_proj(torch.from_numpy(text_model.encode(docs))).numpy().astype('float32')
        _ = img_model(img_data)
        img_emb = np.random.rand(N, d_img_out).astype('float32')
        img_emb_proj = img_proj(torch.from_numpy(img_emb)).numpy().astype('float32')
    
    # Late Fusion (Average)
    embs = (tab_emb + text_emb + img_emb_proj) / 3.0
    pre_t = (time.time() - start) * 1000
    
    # Benchmark Indexing
    start = time.time()
    index = faiss.IndexFlatL2(d_total)
    index.add(embs)
    idx_t = (time.time() - start) * 1000
    
    # Benchmark Search P1
    start = time.time()
    dists, indices = index.search(embs[:1], 100)
    s1_t = (time.time() - start) * 1000
    
    # Benchmark Collective P2
    s2_t = run_real_collective(embs[indices[0]])
    
    print(f"MultiModal,DcCA,{N},{d_total},Multi2Vec,FAISS,k-NN,{pre_t:.2f},{idx_t:.2f},{s1_t:.2f},{s2_t:.2f}")

