from rdkit import Chem
from rdkit.RDLogger import logger
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import AllChem as Chem
import sys
import pandas as pd
import csv


def read_data(file):
    df_raw = pd.read_csv(file)
    df = df_raw[['C_ID', 'SMILES', 'LogS']]
    print(df.head(10))
    return df

file_path = "final_data/final_unique_test.csv"
calc = MolecularDescriptorCalculator(['Chi0' , 'Chi0n' , 'Chi0v' , 'Chi1' , 'Chi1n' , 'Chi1v' , 'Chi2n' , 'Chi2v' , 'Chi3n' , 'Chi3v' , 'Chi4n' , 'Chi4v' , 'EState_VSA1' , 'EState_VSA10' , 'EState_VSA11' , 'EState_VSA2' , 'EState_VSA3' , 'EState_VSA4' , 'EState_VSA5' , 'EState_VSA6' , 'EState_VSA7' , 'EState_VSA8' , 'EState_VSA9' , 'FractionCSP3' , 'HallKierAlpha' , 'HeavyAtomCount' , 'Kappa1' , 'Kappa2' , 'Kappa3' , 'LabuteASA' , 'MolLogP' , 'MolMR' , 'MolWt' , 'NHOHCount' , 'NOCount' , 'NumAliphaticCarbocycles' , 'NumAliphaticHeterocycles' , 'NumAliphaticRings' , 'NumAromaticCarbocycles' , 'NumAromaticHeterocycles' , 'NumAromaticRings' , 'NumHAcceptors' , 'NumHDonors' , 'NumHeteroatoms' , 'NumRotatableBonds' , 'NumSaturatedCarbocycles' , 'NumSaturatedHeterocycles' , 'NumSaturatedRings' , 'PEOE_VSA1' , 'PEOE_VSA10' , 'PEOE_VSA11' , 'PEOE_VSA12' , 'PEOE_VSA13' , 'PEOE_VSA14' , 'PEOE_VSA2' , 'PEOE_VSA3' , 'PEOE_VSA4' , 'PEOE_VSA5' , 'PEOE_VSA6' , 'PEOE_VSA7' , 'PEOE_VSA8' , 'PEOE_VSA9' , 'RingCount' , 'SMR_VSA1' , 'SMR_VSA10' , 'SMR_VSA2' , 'SMR_VSA3' , 'SMR_VSA4' , 'SMR_VSA5' , 'SMR_VSA6' , 'SMR_VSA7' , 'SMR_VSA8' , 'SMR_VSA9' , 'SlogP_VSA1' , 'SlogP_VSA10' , 'SlogP_VSA11' , 'SlogP_VSA12' , 'SlogP_VSA2' , 'SlogP_VSA3' , 'SlogP_VSA4' , 'SlogP_VSA5' , 'SlogP_VSA6' , 'SlogP_VSA7' , 'SlogP_VSA8' , 'SlogP_VSA9' , 'TPSA' , 'VSA_EState1' , 'VSA_EState10' , 'VSA_EState2' , 'VSA_EState3' , 'VSA_EState4' , 'VSA_EState5' , 'VSA_EState6' , 'VSA_EState7' , 'VSA_EState8' , 'VSA_EState9' ])
out_file = open("final_data/reg.csv",'w')

with open(file_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
header = [h.strip("\ufeff ").strip() for h in header]
smiles_col = header.index("smiles_canon") if "smiles_canon" in header else header.index("SMILES")
name_col   = header.index("C_ID")        if "C_ID" in header else -1
# suppl = Chem.SmilesMolSupplier(
#         file_path,
#         delimiter=",",
#         smilesColumn=smiles_col ,
#         nameColumn=name_col   ,
#         titleLine=True        # we have a header row
#     )


df = read_data("final_data/final_unique_test.csv")

nms = list(calc.GetDescriptorNames())
nms2 = ",".join(str(x) for x in nms)
nms2 = 'name,target,' + nms2 + '\n'
out_file.write(nms2)
for _, row in df.iterrows():
    cid = str(row["C_ID"])
    smi = str(row["SMILES"])
    logs = str(row["LogS"])
    mol = Chem.MolFromSmiles(smi)
    descrs = calc.CalcDescriptors(mol)
    descrs2 = ",".join(str(x) for x in descrs)
    descrs2 = str(cid) + ',' + str(logs) + ',' + descrs2 + '\n'
    out_file.write(descrs2)

