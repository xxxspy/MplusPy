from mpluspy.mplus import MplusModel
from mpluspy.output import aggregate_fits
import pandas as pd
import re
import os

def split(names: str):
    match = re.match(r'(?P<prefix>[A-Za-z]+)(?P<start>\d+)-([A-Za-z]+)(?P<end>\d+)', names)  
    if not match:  
        raise ValueError(f"Invalid range string format: {names}")  
      
    prefix = match.group('prefix')  # 提取前缀  
    start_num = int(match.group('start'))  # 提取起始数字  
    end_num = int(match.group('end'))  # 提取结束数字  
      
    # 生成变量名列表  
    variable_names = [f"{prefix}{num:0{len(str(end_num))}d}" for num in range(start_num, end_num + 1)]  
    return variable_names  

igroups = [
    {'name': 'ATT', 'items': split('AT1-AT5')},
    {'name': 'PBCA', 'items': split('PBC1-PBC2')},
    {'name': 'PBCB', 'items': split('PBC3-PBC4')},
    {'name': 'PBC', 'items': 'PBCA,PBCB'.split(',')},
    {'name': 'BI', 'items': split('BI1-BI4')},
    {'name': 'EC', 'items': split('EC1-EC7')},
    {'name': 'SN', 'items': split('SN1-SN5')},
    {'name': 'AB', 'items': split('AB1-AB5')},
    {'name': 'CF', 'items': split('CF1-CF4')},
    {'name': 'PPHL', 'items': split('PPHL1-PPHL4')},
    {'name': 'DPT', 'items': split('DPT1-DPT4')},
    {'name': 'JE', 'items': split('JE1-JE3')},
]

igroups2 = [
    {'name': 'ATI', 'items': split('AT1-AT5')},
    {'name': 'PBCA', 'items': split('PBC1-PBC2')},
    {'name': 'PBCB', 'items': split('PBC3-PBC4')},
    {'name': 'BI', 'items': split('BI1-BI4')},
    {'name': 'EC', 'items': split('EC1-EC7')},
    {'name': 'SN', 'items': split('SN1-SN5')},
    {'name': 'AB', 'items': split('AB1-AB5')},
    {'name': 'CF', 'items': split('CF1-CF4')},
    {'name': 'PPHL', 'items': split('PPHL1-PPHL4')},
    {'name': 'DPT', 'items': split('DPT1-DPT4')},
    {'name': 'JE', 'items': split('JE1-JE3')},
]

def measure_syntax(igroups):
    bystring = ''
    for ig in igroups:
        bystring += f'{ig["name"]} by {" ".join(ig["items"])};\n'
    return bystring

import copy
def merge_igroups(merges: list[tuple]):
    # n: 模型的因子个数
    rtn = []
    for names in merges:
        newigroups = []
        mergename = '_'.join(names)
        mergeitems = []
        for ig in igroups2:
            if ig['name'] in names:
                mergeitems += ig['items']
            else:
                newigroups.append(ig)
        newigroups.append({
            'name': mergename, 
            'items': mergeitems,
        })
        rtn.append(newigroups)
    return rtn
    

# pdata = pd.read_excel(r'D:\dev\notebooks\jobs\医生知识共享患者健康素养和医患关系\data.xlsx')
os.chdir(r'D:\dev\notebooks\jobs\医生知识共享患者健康素养和医患关系')
pdata = pd.read_excel(r'data.xlsx')

# model = MplusModel(
#     TITLE='cfa',
#     MODEL=measure_syntax(igroups),
#     OUTPUT='STANDARDIZED',
#     pdata=pdata,
# )

# print(model.detect_vars_in_model())
# print(model.syntax)
# model.gen_data_file()
# print(model.vnames_in_VARIABLE)
# out = model.fit()

# fitdf = out.model_fit_information_df
# print(fitdf)
# factors = out.factor_loadings
# print(factors)

# ave_cr = out.ave_cr
# print(ave_cr)

# crr = out.corr_matrix
# disdf = out.discriminant_df
# print(disdf)

def factor_merge_fits():
    fits = []
    titles = []
    for igps in merge_igroups([
        ('PBCA', 'PBCB'),
        ('PBCA', 'PBCB', 'ATI'),
        ('PBCA', 'PBCB', 'ATI', 'BI'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN', 'AB'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN', 'AB', 'CF'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN', 'AB', 'CF', 'PPHL'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN', 'AB', 'CF', 'PPHL', 'DPT'),
        ('PBCA', 'PBCB', 'ATI', 'BI', 'EC', 'SN', 'AB', 'CF', 'PPHL', 'DPT', 'JE'),
    ]):
        title = f'{len(igps)}-factors'
        m = MplusModel(
            TITLE=title,
            MODEL=measure_syntax(igps),
            OUTPUT='STANDARDIZED',
            pdata=pdata,
        ).fit()
        fits.append(m.model_fit_information_df)
        titles.append(title)
    df = aggregate_fits(fits, titles)
    return df
        
# fits = factor_merge_fits()
# print(fitdf)
# print(fits)

struc_syntax = '''
    BI on ATT SN PBC;
    AB on CF BI PBC;
    PPHL on AB;
    DPT on AB PPHL;
'''

model = MplusModel(
    TITLE='sem',
    MODEL=measure_syntax(igroups) + struc_syntax,
    OUTPUT=['STANDARDIZED','INDIRECT'],
    MODELINDIRECT=[
        'AB ind ATT',
        'AB ind SN',
        'AB ind PBC',
        'PPHL ind ATT',
        'PPHL ind SN',
        'PPHL ind PBC',
        'DPT ind ATT',
        'DPT ind SN',
        'DPT ind PBC',
    ],
    pdata=pdata,
)

sem = model.fit()
# print(sem.stdyx_model_results_df)
coefs = sem.structure_results_df
print(coefs)
inds = sem.indrect_df
print('%%%%%%%%%%%%%%%%%%%%%%%%')
print(inds)