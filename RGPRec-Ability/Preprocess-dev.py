import random
import numpy as np
import argparse
from enum import IntEnum
import os

class Feature(IntEnum):
    D = 0  # developer ID
    S = 1  # task ID

    CO = 2  # Recommend cooperation
    EF = 3  # Cooperative effect

    DAU = 4  # developer audience
    DST = 5  # developer status
    DPO = 6  # developer position
    DPE = 7  # developer performance
    DLA = 8  # developer language

    SAU = 9  # task audience
    SST = 10  # task status
    STO = 11  # task topic
    SLI = 12  # task licence
    SSY = 13  # task system
    DOU = 14  # task Documents

    
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Preprocess.")
    parser.add_argument('--path', nargs='?', default='../RGPRec-RAG/dataR_result/', help='Input dataA_result path.')
    parser.add_argument('--ratio', type=float, default=0.1, help='Given training ratio.')
    parser.add_argument('--rebuild', type=int, default=0, help='Whether to rebuild whole dataset or not.')
    return parser.parse_args()


def transform_data_format(data_in_tsv_format, data_in_libfm_format):
    print("Transform dataA_result to the format of LibFM...")
    Dset, Sset = set(), set()
    DAUset, DSTset, DPOset, DLAset = set(), set(), set(), set()
    SAUset, SSTset, STOset, SLIset, SSYset, DOUset = set(), set(), set(), set(), set(), set()
    
    reader = open(data_in_tsv_format, 'r')
    output = open(data_in_libfm_format, 'w')
    all_lines = reader.read().splitlines()
    for line in all_lines:
         tmp = line.split("\t")
         Dset.add(tmp[Feature.D])
         Sset.add(tmp[Feature.S])
         DAUset.add(tmp[Feature.DAU])
         DSTset.add(tmp[Feature.DST])
         DPOset.add(tmp[Feature.DPO])
         DLAset.add(tmp[Feature.DLA])
         SAUset.add(tmp[Feature.SAU])
         SSTset.add(tmp[Feature.SST])
         STOset.add(tmp[Feature.STO])
         SLIset.add(tmp[Feature.SLI])
         SSYset.add(tmp[Feature.SSY])
         DOUset.add(tmp[Feature.DOU])
    
    Sset = list(Sset)
    DAUset = list(DAUset)
    DSTset = list(DSTset)
    DPOset = list(DPOset)
    DLAset = list(DLAset)
    SAUset = list(SAUset)
    SSTset = list(SSTset)
    STOset = list(STOset)
    SLIset = list(SLIset)
    SSYset = list(SSYset)
    DOUset = list(DOUset)
         
    for line in all_lines:
         tmp = line.split("\t")
         strCO = tmp[Feature.CO]
         strEF = tmp[Feature.EF]
         strD = tmp[Feature.D]
         strS = str(len(Dset) + Sset.index(tmp[Feature.S]))
         strDAU = str(len(Dset) + len(Sset) + DAUset.index(tmp[Feature.DAU]))
         strDST = str(len(Dset) + len(Sset) + len(DAUset) + DSTset.index(tmp[Feature.DST]))
         strDPO = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + DPOset.index(tmp[Feature.DPO]))
         strDLA = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + DLAset.index((tmp[Feature.DLA])))
         strSAU = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + len(DLAset) + SAUset.index(tmp[Feature.SAU]))
         strSST = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + len(DLAset) + len(SAUset) + SSTset.index(tmp[Feature.SST]))
         strSTO = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + len(DLAset) + len(SAUset) + len(SSTset) + STOset.index(tmp[Feature.STO]))
         strSLI = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + len(DLAset) + len(SAUset) + len(SSTset) + len(STOset) + SLIset.index(tmp[Feature.SLI]))
         strSSY = str(len(Dset) + len(Sset) + len(DAUset) + len(DSTset) + len(DPOset) + len(DLAset) + len(SAUset) + len(SSTset) + len(STOset) + len(SLIset) + SSYset.index(tmp[Feature.SSY]))
         strDOU = tmp[Feature.DOU]
         text = strCO + ' ' + strEF + ' ' + strD + ':' + '1' + ' ' + strS + ':' + '1' + ' ' + strDAU + ':' + '1' + ' ' + strDST + ':' + '1' + ' ' + strDPO + ':' + '1' + ' ' + strDLA + ':' + '1' + ' ' + \
         strSAU + ':' + '1' + ' ' + strSST + ':' + '1' + ' ' + strSTO + ':' + '1' + ' ' + strSLI + ':' + '1' + ' ' + strSSY + ':' + '1' + ' ' + strDOU + '\n'
         #print(text)
         output.write(text)
    reader.close()
    output.close()
    print('Done!')


def max_min_normalize(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x;


def split_into_train_test(input_data, ratio, train_data, test_data, seed=2021):
    print(f"Split {input_data} into {train_data} and {test_data} with ratio: {ratio}...")
    input = open(input_data, 'r')
    outtr = open(train_data, 'w')
    outte = open(test_data, 'w')
    
    all_lines = input.read().splitlines()
    all_lines = list(all_lines)
    
    np.random.seed(seed)
    random.shuffle(all_lines)
    print ('all_lines:',len(all_lines))

    train_samples = all_lines[:int(ratio * len(all_lines))] 
    
    for line in train_samples:
          outtr.write(line + '\n')

    test_samples = all_lines[int(ratio * len(all_lines)):]
    
    for line in test_samples:
          outte.write(line + '\n')    
    
    input.close()
    outtr.close()
    outte.close()
    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    ratio = args.ratio
    if args.rebuild == 1:
        transform_data_format(path + '/RGPRecA_Context.dat', path + '/RGPRecA.libfm')
    subdir = path + str(int(ratio * 100))
    #===========================================================================
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    #===========================================================================
    split_into_train_test(path + '/RGPRecA.libfm', ratio, subdir + '/RGPRecA_train.libfm', subdir + '/RGPRecA_test.libfm')
