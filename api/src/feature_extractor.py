from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
# from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
#
# import pandas as pd
# import numpy as np
# import feature_extractor as fe
# from rdkit import Chem
# from rdkit.Chem import PandasTools
# from rdkit import DataStructs
#
# from rdkit.Chem import PandasTools
#
# import pandas_profiling as pp
#
# # Display setting
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
#
# def data_for_models(data):
#     data_4_model = pd.read_csv(data)
#     return data_4_model
#
#
def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )
#
# data4_M1_M2 = data_for_models("./dataset_single.csv")
# data4_M3 = data_for_models("./dataset_multi.csv")


# Data Exploration
#
# report_data4_M1_M2 = pp.ProfileReport(data4_M1_M2)
# report_data4_M1_M2.to_file('profile_report_data4_M1_M2.html')
#
# report_data4_M3 = pp.ProfileReport(data4_M3)
# report_data4_M3.to_file('profile_report_data4_M3.html')


#
# # extract feature
# data4_M1_M2['smiles_features'] = data4_M1_M2.smiles.apply(lambda x: np.array(fe.fingerprint_features(x)))
# data4_M3['smiles_features'] = data4_M3.smiles.apply(lambda x: np.array(fe.fingerprint_features(x)))
#
#
# print(data4_M1_M2)
# print('*'*100)
# print(data4_M3)
#
#
# exit()
#
# feature_vector = fingerprint_features('Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C')
#
# print(feature_vector.ToBitString())
# print(feature_vector.ToBitString().count("1"))
# print(feature_vector.ToBitString().count("0"))
# print(len(feature_vector.ToBitString()))
#
# print('*'*100)
#
# feature_vector_2 = fingerprint_features('CCn1c(CSc2nccn2C)nc2cc(C(=O)O)ccc21')
#
# print(feature_vector_2.ToBitString())
# print(feature_vector_2.ToBitString().count("1"))
# print(feature_vector_2.ToBitString().count("0"))
# print(len(feature_vector_2.ToBitString()))