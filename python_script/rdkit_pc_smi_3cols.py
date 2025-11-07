"""
Copyright (c) 2023 Ulf Norinder

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from rdkit import Chem
from rdkit.RDLogger import logger
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import AllChem as Chem
import sys
logger=logger()
#import cPickle


try:
    sys.argv[1]
except IndexError:
    print ("You need to specify an input file (smi, ID, label)")
    sys.exit(1)

try:
    sys.argv[2]
except IndexError:
    print ("You need to specify format: smiles id (s), id target smiles (d), smiles label (s2) or smiles id target (d2)")
    sys.exit(1)

try:
    sys.argv[3]
except IndexError:
    print ("You need to specify separator: space(s), comma (c) or tab (t)")
    sys.exit(1)

if sys.argv[3] == 's':
    sep = 's'
    sep2 = ' '
if sys.argv[3] == 't':
    sep = 't'
    sep2 = '\t'

if sys.argv[3] == 'c':
    sep = 'c'
    sep2 = ','

#calc = MolecularDescriptorCalculator(['MolLogP','NOCount','NHOHCount','MolWt','NumRotatableBonds','TPSA','Chi0' , 'Chi0n' , 'Chi0v' , 'Chi1' , 'Chi1n' , 'Chi1v' , 'Chi2n' , 'Chi2v' , 'Chi3n' , 'Chi3v' , 'Chi4n' , 'Chi4v'])

#calc = MolecularDescriptorCalculator(['Chi0' , 'Chi0n' , 'Chi0v' , 'Chi1' , 'Chi1n' , 'Chi1v' , 'Chi2n' , 'Chi2v' , 'Chi3n' , 'Chi3v' , 'Chi4n' , 'Chi4v' , 'EState_VSA1' , 'EState_VSA10' , 'EState_VSA11' , 'EState_VSA2' , 'EState_VSA3' , 'EState_VSA4' , 'EState_VSA5' , 'EState_VSA6' , 'EState_VSA7' , 'EState_VSA8' , 'EState_VSA9' , 'FractionCSP3' , 'HallKierAlpha' , 'HeavyAtomCount' , 'Ipc' , 'Kappa1' , 'Kappa2' , 'Kappa3' , 'LabuteASA' , 'MolLogP' , 'MolMR' , 'NHOHCount' , 'NOCount' , 'NumAliphaticCarbocycles' , 'NumAliphaticHeterocycles' , 'NumAliphaticRings' , 'NumAromaticCarbocycles' , 'NumAromaticHeterocycles' , 'NumAromaticRings' , 'NumHAcceptors' , 'NumHDonors' , 'NumHeteroatoms' , 'NumRotatableBonds' , 'NumSaturatedCarbocycles' , 'NumSaturatedHeterocycles' , 'NumSaturatedRings' , 'PEOE_VSA1' , 'PEOE_VSA10' , 'PEOE_VSA11' , 'PEOE_VSA12' , 'PEOE_VSA13' , 'PEOE_VSA14' , 'PEOE_VSA2' , 'PEOE_VSA3' , 'PEOE_VSA4' , 'PEOE_VSA5' , 'PEOE_VSA6' , 'PEOE_VSA7' , 'PEOE_VSA8' , 'PEOE_VSA9' , 'RingCount' , 'SMR_VSA1' , 'SMR_VSA10' , 'SMR_VSA2' , 'SMR_VSA3' , 'SMR_VSA4' , 'SMR_VSA5' , 'SMR_VSA6' , 'SMR_VSA7' , 'SMR_VSA8' , 'SMR_VSA9' , 'SlogP_VSA1' , 'SlogP_VSA10' , 'SlogP_VSA11' , 'SlogP_VSA12' , 'SlogP_VSA2' , 'SlogP_VSA3' , 'SlogP_VSA4' , 'SlogP_VSA5' , 'SlogP_VSA6' , 'SlogP_VSA7' , 'SlogP_VSA8' , 'SlogP_VSA9' , 'TPSA' , 'VSA_EState1' , 'VSA_EState10' , 'VSA_EState2' , 'VSA_EState3' , 'VSA_EState4' , 'VSA_EState5' , 'VSA_EState6' , 'VSA_EState7' , 'VSA_EState8' , 'VSA_EState9' , 'fr_Al_COO' , 'fr_Al_OH' , 'fr_Al_OH_noTert' , 'fr_ArN' , 'fr_Ar_COO' , 'fr_Ar_N' , 'fr_Ar_NH' , 'fr_Ar_OH' , 'fr_COO' , 'fr_COO2' , 'fr_C_O' , 'fr_C_O_noCOO' , 'fr_C_S' , 'fr_HOCCN' , 'fr_Imine' , 'fr_NH0' , 'fr_NH1' , 'fr_NH2' , 'fr_N_O' , 'fr_Ndealkylation1' , 'fr_Ndealkylation2' , 'fr_Nhpyrrole' , 'fr_SH' , 'fr_aldehyde' , 'fr_alkyl_carbamate' , 'fr_alkyl_halide' , 'fr_allylic_oxid' , 'fr_amide' , 'fr_amidine' , 'fr_aniline' , 'fr_aryl_methyl' , 'fr_azide' , 'fr_azo' , 'fr_barbitur' , 'fr_benzene' , 'fr_benzodiazepine' , 'fr_bicyclic' , 'fr_diazo' , 'fr_dihydropyridine' , 'fr_epoxide' , 'fr_ester' , 'fr_ether' , 'fr_furan' , 'fr_guanido' , 'fr_halogen' , 'fr_hdrzine' , 'fr_hdrzone' , 'fr_imidazole' , 'fr_imide' , 'fr_isocyan' , 'fr_isothiocyan' , 'fr_ketone' , 'fr_ketone_Topliss' , 'fr_lactam' , 'fr_lactone' , 'fr_methoxy' , 'fr_morpholine' , 'fr_nitrile' , 'fr_nitro' , 'fr_nitro_arom' , 'fr_nitro_arom_nonortho' , 'fr_nitroso' , 'fr_oxazole' , 'fr_oxime' , 'fr_para_hydroxylation' , 'fr_phenol' , 'fr_phenol_noOrthoHbond' , 'fr_phos_acid' , 'fr_phos_ester' , 'fr_piperdine' , 'fr_piperzine' , 'fr_priamide' , 'fr_prisulfonamd' , 'fr_pyridine' , 'fr_quatN' , 'fr_sulfide' , 'fr_sulfonamd' , 'fr_sulfone' , 'fr_term_acetylene' , 'fr_tetrazole' , 'fr_thiazole' , 'fr_thiocyan' , 'fr_thiophene' , 'fr_unbrch_alkane' , 'fr_urea'])

#calc = MolecularDescriptorCalculator(['Chi0' , 'Chi0n' , 'Chi0v' , 'Chi1' , 'Chi1n' , 'Chi1v' , 'Chi2n' , 'Chi2v' , 'Chi3n' , 'Chi3v' , 'Chi4n' , 'Chi4v' , 'EState_VSA1' , 'EState_VSA10' , 'EState_VSA11' , 'EState_VSA2' , 'EState_VSA3' , 'EState_VSA4' , 'EState_VSA5' , 'EState_VSA6' , 'EState_VSA7' , 'EState_VSA8' , 'EState_VSA9' , 'FractionCSP3' , 'HallKierAlpha' , 'HeavyAtomCount' , 'Ipc' , 'Kappa1' , 'Kappa2' , 'Kappa3' , 'LabuteASA' , 'MolLogP' , 'MolMR' , 'MolWt' , 'NHOHCount' , 'NOCount' , 'NumAliphaticCarbocycles' , 'NumAliphaticHeterocycles' , 'NumAliphaticRings' , 'NumAromaticCarbocycles' , 'NumAromaticHeterocycles' , 'NumAromaticRings' , 'NumHAcceptors' , 'NumHDonors' , 'NumHeteroatoms' , 'NumRotatableBonds' , 'NumSaturatedCarbocycles' , 'NumSaturatedHeterocycles' , 'NumSaturatedRings' , 'PEOE_VSA1' , 'PEOE_VSA10' , 'PEOE_VSA11' , 'PEOE_VSA12' , 'PEOE_VSA13' , 'PEOE_VSA14' , 'PEOE_VSA2' , 'PEOE_VSA3' , 'PEOE_VSA4' , 'PEOE_VSA5' , 'PEOE_VSA6' , 'PEOE_VSA7' , 'PEOE_VSA8' , 'PEOE_VSA9' , 'RingCount' , 'SMR_VSA1' , 'SMR_VSA10' , 'SMR_VSA2' , 'SMR_VSA3' , 'SMR_VSA4' , 'SMR_VSA5' , 'SMR_VSA6' , 'SMR_VSA7' , 'SMR_VSA8' , 'SMR_VSA9' , 'SlogP_VSA1' , 'SlogP_VSA10' , 'SlogP_VSA11' , 'SlogP_VSA12' , 'SlogP_VSA2' , 'SlogP_VSA3' , 'SlogP_VSA4' , 'SlogP_VSA5' , 'SlogP_VSA6' , 'SlogP_VSA7' , 'SlogP_VSA8' , 'SlogP_VSA9' , 'TPSA' , 'VSA_EState1' , 'VSA_EState10' , 'VSA_EState2' , 'VSA_EState3' , 'VSA_EState4' , 'VSA_EState5' , 'VSA_EState6' , 'VSA_EState7' , 'VSA_EState8' , 'VSA_EState9' ])
calc = MolecularDescriptorCalculator(['Chi0' , 'Chi0n' , 'Chi0v' , 'Chi1' , 'Chi1n' , 'Chi1v' , 'Chi2n' , 'Chi2v' , 'Chi3n' , 'Chi3v' , 'Chi4n' , 'Chi4v' , 'EState_VSA1' , 'EState_VSA10' , 'EState_VSA11' , 'EState_VSA2' , 'EState_VSA3' , 'EState_VSA4' , 'EState_VSA5' , 'EState_VSA6' , 'EState_VSA7' , 'EState_VSA8' , 'EState_VSA9' , 'FractionCSP3' , 'HallKierAlpha' , 'HeavyAtomCount' , 'Kappa1' , 'Kappa2' , 'Kappa3' , 'LabuteASA' , 'MolLogP' , 'MolMR' , 'MolWt' , 'NHOHCount' , 'NOCount' , 'NumAliphaticCarbocycles' , 'NumAliphaticHeterocycles' , 'NumAliphaticRings' , 'NumAromaticCarbocycles' , 'NumAromaticHeterocycles' , 'NumAromaticRings' , 'NumHAcceptors' , 'NumHDonors' , 'NumHeteroatoms' , 'NumRotatableBonds' , 'NumSaturatedCarbocycles' , 'NumSaturatedHeterocycles' , 'NumSaturatedRings' , 'PEOE_VSA1' , 'PEOE_VSA10' , 'PEOE_VSA11' , 'PEOE_VSA12' , 'PEOE_VSA13' , 'PEOE_VSA14' , 'PEOE_VSA2' , 'PEOE_VSA3' , 'PEOE_VSA4' , 'PEOE_VSA5' , 'PEOE_VSA6' , 'PEOE_VSA7' , 'PEOE_VSA8' , 'PEOE_VSA9' , 'RingCount' , 'SMR_VSA1' , 'SMR_VSA10' , 'SMR_VSA2' , 'SMR_VSA3' , 'SMR_VSA4' , 'SMR_VSA5' , 'SMR_VSA6' , 'SMR_VSA7' , 'SMR_VSA8' , 'SMR_VSA9' , 'SlogP_VSA1' , 'SlogP_VSA10' , 'SlogP_VSA11' , 'SlogP_VSA12' , 'SlogP_VSA2' , 'SlogP_VSA3' , 'SlogP_VSA4' , 'SlogP_VSA5' , 'SlogP_VSA6' , 'SlogP_VSA7' , 'SlogP_VSA8' , 'SlogP_VSA9' , 'TPSA' , 'VSA_EState1' , 'VSA_EState10' , 'VSA_EState2' , 'VSA_EState3' , 'VSA_EState4' , 'VSA_EState5' , 'VSA_EState6' , 'VSA_EState7' , 'VSA_EState8' , 'VSA_EState9' ])

if sys.argv[2] == 's':
    suppl = Chem.SmilesMolSupplier(sys.argv[1],delimiter=sep2,smilesColumn = 0,nameColumn=1,titleLine=False)
if sys.argv[2] == 's2':
    suppl = Chem.SmilesMolSupplier(sys.argv[1],delimiter=sep2,smilesColumn = 0,nameColumn=1,titleLine=False)
if sys.argv[2] == 'd':
    suppl = Chem.SmilesMolSupplier(sys.argv[1],delimiter=sep2,smilesColumn = 2,nameColumn=0,titleLine=False)
if sys.argv[2] == 'd2':
    suppl = Chem.SmilesMolSupplier(sys.argv[1],delimiter=sep2,smilesColumn = 0,nameColumn=1,titleLine=False)

#bb =sys.argv[1]+".rdkit"
bb =sys.argv[1]+".rdkit.txt"
#w = Chem.SmilesWriter(bb)
#w.SetProps(nms)
f2 = open(bb,'w')

nms = list(calc.GetDescriptorNames())
nms2 = "\t".join(str(x) for x in nms)
nms2 = 'name\ttarget\t' + nms2 + '\n'
f2.write(nms2)

nDone=0
pflag = 0
f = open(sys.argv[1],'r')
for mol in suppl:
    line = f.readline()
    line = line.strip()
    if sys.argv[2] == 's':
        smi, ID = line.split(sep2)
    if sys.argv[2] == 's2':
        smi, label = line.split(sep2)
        ID = smi
    if sys.argv[2] == 'd':
        try:
            ID, target, smi = line.split(sep2)
        except:
            if nDone < 1:
                print ("Trying 4 columns but keeping only the first 3: ID, target, smi")
            ID, target, smi, dummy = line.split(sep2)
    if sys.argv[2] == 'd2':
        smi, ID, target = line.split(sep2)
    nDone += 1
    line = line.strip()
    if not nDone%1000: logger.info("Done %d"%nDone)
#    if mol is None: break
    try:
        if mol.GetNumAtoms():
            descrs = calc.CalcDescriptors(mol)
#            for nm,v in zip(nms,descrs):
#                 mol.SetProp(nm,str(v))
#            w.write(mol)
            descrs2 = "\t".join(str(x) for x in descrs)
            try:
                descrs2 = str(ID) + '\t' + str(label) + '\t' + descrs2 + '\n'
            except:
                if pflag == 0:
                    print ("Trying 'target' insted of 'label'\n")
                    pflag = 1
                descrs2 = str(ID) + '\t' + str(target) + '\t' + descrs2 + '\n'
            f2.write(descrs2)
    except AttributeError:
        print ("Error",mol)

