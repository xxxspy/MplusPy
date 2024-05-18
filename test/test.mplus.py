from mpluspy.mplus import MplusModel
import pandas as pd
import re

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
    {'name': 'AT', 'items': split('AT1-AT5')},
    {'name': 'PBC', 'items': split('PBC1-PBC4')},
    {'name': 'BI', 'items': split('BI1-BI4')},
    {'name': 'EC', 'items': split('EC1-EC7')},
    {'name': 'SN', 'items': split('SN1-SN5')},
    {'name': 'AB', 'items': split('AB1-AB5')},
    {'name': 'CF', 'items': split('CF1-CF4')},
    {'name': 'PPHL', 'items': split('PPHL1-PPHL4')},
    {'name': 'DPT', 'items': split('DPT1-DPT4')},
    {'name': 'JE', 'items': split('JE1-JE3')},
]

bystring = ''
for ig in igroups:
    bystring += f'{ig["name"]} by {" ".join(ig["items"])};\n'

print(igroups)

# pdata = pd.read_excel(r'D:\dev\notebooks\jobs\医生知识共享患者健康素养和医患关系\data.xlsx')

model = MplusModel(
    TITLE='验证性因子分析',
    MODEL=bystring,
    OUTPUT='stdxy',
    # pdata=pdata,
)

print(model.detect_vars_in_model())
print(model.syntax)
# model.gen_data_file()
print(model.vnames_in_VARIABLE)