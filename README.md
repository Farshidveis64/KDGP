KDGP: Knowledge Distillation for GNN Pruning]{An Effective Knowledge Distillation Framework for Graph Neural Network Pruning
Official code for the ACML 2026 submission *(under double-blind review)*.

FastGLT finds sparse, trainable subnetworks of a graph neural network by
**jointly sparsifying model weights and graph adjacency** during a single
grow-and-prune schedule, while a dense **teacher** guides the sparse student
through knowledge distillation. The method is implemented over three backbones
— **GCN**, **GAT**, and **GIN** — and evaluated on the standard Planetoid
citation benchmarks (Cora, Citeseer, Pubmed).

---

## Repository structure

```
.
├── GCN/                # GCN backbone
│   ├── main.py         # training entry point (pretrain → prune → rewind+retrain)
│   ├── net.py          # sparse GCN model (weight + adjacency masks)
│   ├── layer_gcn.py    # GCN layer
│   ├── pruning.py      # FastScheduler: grow-and-cut sparsity scheduler
│   ├── utils.py        # data loading, seeding, mask utilities, losses
│   ├── normalization.py
│   ├── args.py         # command-line arguments and per-dataset config
│   └── data/           # Planetoid files (ind.{cora,citeseer,pubmed}.*)
├── GAT/                # GAT backbone (same layout; net_gat.py, layer_gat.py)
├── GIN/                # GIN backbone (same layout; net_gin.py, layer_gin.py)
├── requirements.txt
├── LICENSE
└── README.md
```

Each backbone directory is **self-contained** and uses flat, local imports
(`import net`, `import utils`, `from args import parser_loader`). Always run a
backbone from **inside its own directory** so the imports and the bundled
`data/` folder resolve correctly.

---

## Installation

Tested with **Python 3.9**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `dgl`, and `torch-geometric` should be installed with the build that
matches your CUDA toolkit; see the official installation guides for each. A
CUDA-capable GPU is expected: the training loop moves features and labels to
the GPU (`.cuda()`), so a GPU runtime is required to run the code as provided.

---

## Data

The Planetoid citation datasets (Cora, Citeseer, Pubmed) are included under each
backbone's `data/` directory in the standard `ind.<dataset>.*` format, so no
separate download is needed.

---

## Pretrained teacher (required for KD)

The distillation step loads a dense teacher checkpoint from `args['teacher_path']`
(set near the bottom of each `main.py`). Train or supply a dense model of the
same backbone/dataset and point `teacher_path` at its `state_dict` before
running. The relevant knobs are also set there:

```python
args['teacher_path'] = 'teacher_model_new.pth'  # dense teacher state_dict
args['temp']         = 1.0                       # KD temperature
args['kd_lambda']    = 0.6                       # KD loss weight
```

---

## Usage

Run each backbone from its own directory. Example (GCN on Cora):

```bash
cd GCN
python main.py --dataset cora --device cuda \
    --pretrain_epoch 50 --total_epoch 400 --retrain_epoch 200 \
    --remain 0.05 --spar_wei --spar_adj
```

GAT and GIN follow the same interface:

```bash
cd GAT && python main.py --dataset citeseer --device cuda --remain 0.05
cd GIN && python main.py --dataset pubmed   --device cuda --remain 0.05
```

The run prints per-epoch loss, train/val/test micro-F1, current weight
sparsity (`WS`) and adjacency sparsity (`AS`), and the best sparse subnetwork
that meets the target density.

---

## Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `cora` | `cora`, `citeseer`, or `pubmed` |
| `--device` | `cpu` | set to `cuda` to train on GPU |
| `--pretrain_epoch` | `50` | dense warm-up epochs before pruning |
| `--total_epoch` | `400` | epochs of the grow-and-prune (KD) phase |
| `--retrain_epoch` | `200` | epochs to retrain the rewound sparse subnetwork |
| `--remain` | `1.0` | target fraction of parameters/edges to keep, in (0, 1] |
| `--spar_wei` | `True` | sparsify model weights |
| `--spar_adj` | `True` | sparsify graph adjacency |
| `--lr` | `0.001` | learning rate |
| `--weight-decay` | `5e-4` | weight decay |
| `--num_layers` | `2` | number of GNN layers |
| `--coef` | `0.1` | mask regularization coefficient |
| `--delta` | `20` | scheduler update interval |
| `--alpha` | `0.3` | scheduler pruning rate |

The full list is in each backbone's `args.py`.

---

## Reproducibility

Per-dataset random seeds are fixed in `args.py` (`cora: 1899`,
`citeseer: 17889`, `pubmed: 3333`) and applied via `utils.fix_seed`. The saved
experiment directory (`CKPTs/`) snapshots the source files used for each run.

---

## License

Released under the MIT License; see [LICENSE](LICENSE).

## Citation

> Anonymous Author(s). *FastGLT: Fast Graph Lottery Tickets with Knowledge
> Distillation.* Under review at ACML 2026. A citation entry will be added upon
> acceptance.
