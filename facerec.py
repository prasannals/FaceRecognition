import os
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict, Optional
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", ctx_id=0):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self.app.get(img)
        if not faces:
            return None
        return faces[0].embedding

    def embed_images_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        return [self.embed_image(p) for p in image_paths]


class EmbeddingDatabase:
    def __init__(self, embedder: FaceEmbedder):
        self.embedder = embedder
        self.entries = []

    def build_from_directory(self, dir_path: str):
        for label in os.listdir(dir_path):
            label_path = os.path.join(dir_path, label)
            if not os.path.isdir(label_path):
                continue
            image_paths = glob.glob(os.path.join(label_path, '*'))
            for path in tqdm(image_paths, desc=f"Embedding {label}"):
                emb = self.embedder.embed_image(path)
                if emb is not None:
                    self.entries.append({
                        "embedding": emb.tolist(),
                        "label": label,
                        "image_path": path
                    })

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.entries, f)

    def load_from_file(self, path: str):
        with open(path, 'r') as f:
            self.entries = json.load(f)


class Evaluator:
    def __init__(self, train_embeddings: List[Dict], k_values=[1, 3, 5]):
        self.train_embeddings = train_embeddings
        self.k_values = k_values

    def _compute_similarity(self, emb1, emb2):
        return 1 - cosine(emb1, emb2)

    def evaluate(self, eval_embeddings: List[Dict], ground_truth: Dict) -> Dict:
        results = {}
        hits = {str(k): 0 for k in self.k_values}

        for entry in tqdm(eval_embeddings, desc="Evaluating"):
            query_emb = np.array(entry["embedding"])
            query_name = os.path.basename(entry["image_path"])
            true_label = ground_truth.get(query_name, None)

            sims = []
            for train_entry in self.train_embeddings:
                sim = self._compute_similarity(query_emb, np.array(train_entry["embedding"]))
                sims.append((sim, train_entry["label"], train_entry["image_path"]))

            sims.sort(reverse=True)
            top_k = sims[:max(self.k_values)]
            top_labels = [label for _, label, _ in top_k]

            result = {
                "ground_truth": true_label,
                "top_k": [
                    {"label": label, "score": float(score), "image": path}
                    for score, label, path in top_k
                ],
                "hits": {}
            }

            for k in self.k_values:
                result["hits"][str(k)] = true_label in top_labels[:k]
                if result["hits"][str(k)]:
                    hits[str(k)] += 1

            results[query_name] = result

        summary = {
            f"hits@{k}": hits[str(k)] / len(eval_embeddings)
            for k in self.k_values
        }
        summary["num_eval_images"] = len(eval_embeddings)

        return {"per_image": results, "summary": summary}

    def save_results(self, result_dict: Dict, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "hits_at_k.json"), "w") as f:
            json.dump(result_dict["per_image"], f, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(result_dict["summary"], f, indent=2)


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--eval_labels", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--ctx_id", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--reuse_embeddings", action="store_true")
    args = parser.parse_args()

    embed_dir = "embeddings"
    os.makedirs(embed_dir, exist_ok=True)

    embedder = FaceEmbedder(ctx_id=args.ctx_id)

    # Train embeddings
    train_db = EmbeddingDatabase(embedder)
    train_embed_path = os.path.join(embed_dir, "train_embeddings.json")
    if args.reuse_embeddings and os.path.exists(train_embed_path):
        train_db.load_from_file(train_embed_path)
    else:
        train_db.build_from_directory(args.train_dir)
        train_db.save_to_file(train_embed_path)

    # Eval embeddings
    eval_db = EmbeddingDatabase(embedder)
    eval_embed_path = os.path.join(embed_dir, "eval_embeddings.json")
    if args.reuse_embeddings and os.path.exists(eval_embed_path):
        eval_db.load_from_file(eval_embed_path)
    else:
        image_paths = glob.glob(os.path.join(args.eval_dir, '*'))
        for path in tqdm(image_paths, desc="Embedding eval"):
            emb = embedder.embed_image(path)
            if emb is not None:
                eval_db.entries.append({
                    "embedding": emb.tolist(),
                    "image_path": path
                })
        eval_db.save_to_file(eval_embed_path)

    # Evaluation
    with open(args.eval_labels, 'r') as f:
        gt = json.load(f)

    evaluator = Evaluator(train_embeddings=train_db.entries, k_values=args.k)
    result = evaluator.evaluate(eval_embeddings=eval_db.entries, ground_truth=gt)
    evaluator.save_results(result, args.out_dir)