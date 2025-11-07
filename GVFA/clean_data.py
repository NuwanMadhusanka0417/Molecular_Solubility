# drop_bad_molecules.py
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# small helpers to improve sanitize robustness (optional but useful)
_NITRO   = Chem.MolFromSmarts('[N;X3](=O)=O')
_NITRO_OK= Chem.MolFromSmiles('[N+](=O)[O-]')

def _drop_orphan_H(m):
    to_del = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==1 and a.GetDegree()==0]
    if not to_del: return m
    em = Chem.EditableMol(m)
    for i in sorted(to_del, reverse=True): em.RemoveAtom(i)
    return em.GetMol()

def _largest_by_bonds(m):
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not fr: return m
    return max(fr, key=lambda f: (f.GetNumBonds(), f.GetNumAtoms()))

def _clean_and_sanitize(smi: str):
    m = Chem.MolFromSmiles(str(smi), sanitize=False)
    if m is None: return None
    m = _drop_orphan_H(m)
    try:
        reps = Chem.ReplaceSubstructs(m, _NITRO, _NITRO_OK, replaceAll=True)
        if reps and reps[0] is not None: m = reps[0]
    except Exception:
        pass
    m = _largest_by_bonds(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

def make_fixed_csv(in_csv: str,
                   out_csv: str,
                   dropped_report_csv: str = None,
                   smiles_col: str = "smiles_canon"):
    df = pd.read_csv(in_csv)
    keep_idx = []
    dropped  = []

    for idx, smi in df[smiles_col].astype(str).items():
        m = _clean_and_sanitize(smi)
        if m is None:
            dropped.append({"row_index": idx, "smiles": smi, "reason": "sanitize_failed"})
            continue
        if m.GetNumBonds() == 0:
            reason = "single_atom_or_zero_bond"
            dropped.append({"row_index": idx, "smiles": smi, "reason": reason})
            continue
        keep_idx.append(idx)

    df_fixed = df.loc[keep_idx].reset_index(drop=True)
    df_fixed.to_csv(out_csv, index=False)
    print(f"[INFO] Kept {len(df_fixed)} / {len(df)} → wrote {out_csv}")

    if dropped_report_csv:
        pd.DataFrame(dropped).to_csv(dropped_report_csv, index=False)
        print(f"[INFO] Dropped {len(dropped)} rows → {dropped_report_csv}")


make_fixed_csv(
    in_csv="final_data/final_unique_train.csv",
    out_csv="final_data/final_unique_train_fixed.csv",
    dropped_report_csv="final_data/train_dropped_rows.csv",
    smiles_col="smiles_canon"
)