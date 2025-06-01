import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict, Optional
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
from itertools import combinations


class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", ctx_id=0):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(str(image_path))
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
        dir_path = Path(dir_path)
        for label_dir in dir_path.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            image_paths = list(label_dir.glob("*"))
            for path in tqdm(image_paths, desc=f"Embedding {label}"):
                emb = self.embedder.embed_image(path)
                if emb is not None:
                    self.entries.append(
                        {
                            "embedding": emb.tolist(),
                            "label": label,
                            "image_path": str(path),
                        }
                    )

    def compute_intra_class_thresholds(self, percentile=5) -> Dict[str, float]:
        thresholds = {}
        label_to_embs = {}

        for entry in self.entries:
            label = entry["label"]
            emb = np.array(entry["embedding"])
            label_to_embs.setdefault(label, []).append(emb)

        for label, embs in label_to_embs.items():
            if len(embs) < 2:
                thresholds[label] = -1
                continue
            sims = [1 - cosine(a, b) for a, b in combinations(embs, 2)]
            thresholds[label] = float(np.percentile(sims, percentile))

        return thresholds

    def save_to_file(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.entries, f)

    def load_from_file(self, path: str):
        path = Path(path)
        with path.open("r") as f:
            self.entries = json.load(f)


class Evaluator:
    def __init__(
        self,
        train_embeddings: List[Dict],
        k_values=[1, 3, 5],
        class_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.train_embeddings = train_embeddings
        self.k_values = k_values
        self.class_thresholds = class_thresholds or {}

    def _compute_similarity(self, emb1, emb2):
        return 1 - cosine(emb1, emb2)

    def evaluate(self, eval_embeddings: List[Dict], ground_truth: Dict) -> Dict:
        results = {}
        hits = {str(k): 0 for k in self.k_values}

        for entry in tqdm(eval_embeddings, desc="Evaluating"):
            query_emb = np.array(entry["embedding"])
            query_name = Path(entry["image_path"]).name
            true_label = ground_truth.get(query_name, None)

            sims = []
            for train_entry in self.train_embeddings:
                label = train_entry["label"]
                threshold = self.class_thresholds.get(label, -1)
                sim = self._compute_similarity(
                    query_emb, np.array(train_entry["embedding"])
                )
                if sim >= threshold:
                    sims.append((sim, label, train_entry["image_path"]))

            sims.sort(reverse=True)
            top_k = sims[: max(self.k_values)]
            top_labels = [label for _, label, _ in top_k]

            result = {
                "ground_truth": true_label,
                "top_k": [
                    {"label": label, "score": float(score), "image": path}
                    for score, label, path in top_k
                ],
                "hits": {},
                "filtered_due_to_threshold": len(sims) < len(self.train_embeddings),
            }

            for k in self.k_values:
                result["hits"][str(k)] = true_label in top_labels[:k]
                if result["hits"][str(k)]:
                    hits[str(k)] += 1

            results[query_name] = result

        summary = {
            f"hits@{k}": hits[str(k)] / len(eval_embeddings) for k in self.k_values
        }
        summary["num_eval_images"] = len(eval_embeddings)

        return {"per_image": results, "summary": summary}

    def save_results(self, result_dict: Dict, out_dir: str):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "hits_at_k.json").open("w") as f:
            json.dump(result_dict["per_image"], f, indent=2)
        with (out_dir / "summary.json").open("w") as f:
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
    parser.add_argument("--use_class_thresholds", action="store_true", default=True)
    args = parser.parse_args()

    embed_dir = Path("embeddings")
    embed_dir.mkdir(parents=True, exist_ok=True)

    embedder = FaceEmbedder(ctx_id=args.ctx_id)

    # Train embeddings
    train_db = EmbeddingDatabase(embedder)
    train_embed_path = embed_dir / "train_embeddings.json"
    if args.reuse_embeddings and train_embed_path.exists():
        train_db.load_from_file(train_embed_path)
    else:
        train_db.build_from_directory(args.train_dir)
        train_db.save_to_file(train_embed_path)

    # Compute class thresholds
    class_thresholds = {}
    thresholds_path = embed_dir / "class_thresholds.json"
    if args.use_class_thresholds:
        if thresholds_path.exists():
            with thresholds_path.open("r") as f:
                class_thresholds = json.load(f)
        else:
            class_thresholds = train_db.compute_intra_class_thresholds()
            with thresholds_path.open("w") as f:
                json.dump(class_thresholds, f, indent=2)

    # Eval embeddings
    eval_db = EmbeddingDatabase(embedder)
    eval_embed_path = embed_dir / "eval_embeddings.json"
    if args.reuse_embeddings and eval_embed_path.exists():
        eval_db.load_from_file(eval_embed_path)
    else:
        image_paths = list(Path(args.eval_dir).glob("*"))
        for path in tqdm(image_paths, desc="Embedding eval"):
            emb = embedder.embed_image(path)
            if emb is not None:
                eval_db.entries.append(
                    {"embedding": emb.tolist(), "image_path": str(path)}
                )
        eval_db.save_to_file(eval_embed_path)

    # Evaluation
    with Path(args.eval_labels).open("r") as f:
        gt = json.load(f)

    evaluator = Evaluator(
        train_embeddings=train_db.entries,
        k_values=args.k,
        class_thresholds=class_thresholds,
    )
    result = evaluator.evaluate(eval_embeddings=eval_db.entries, ground_truth=gt)
    evaluator.save_results(result, args.out_dir)
