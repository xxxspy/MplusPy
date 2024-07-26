from pathlib import Path
import re
from typing import List
import pandas as pd
from functools import cached_property

class MplusError(ValueError):
    pass

VAR_NAME_PTN = r'[a-zA-Z0-9\_]+'
NUMBER_PTN = r'\s([\-0-9\.]+)'


class MplusParser:
    TITLES = ("MODEL RESULTS", "STANDARDIZED MODEL RESULTS", "MODEL FIT INFORMATION",
              'TOTAL, TOTAL INDIRECT, SPECIFIC INDIRECT, AND DIRECT EFFECTS', 
              'STANDARDIZED TOTAL, TOTAL INDIRECT, SPECIFIC INDIRECT, AND DIRECT EFFECTS')
    
    def __init__(self, fpath: str) -> None:
        self.fpath = fpath
        content = Path(fpath).read_text(encoding='utf8')
        title_ptn = r"^\n\n[A-Z][A-Z,\d \-]*\n$"
        titles = re.findall(title_ptn, content, re.MULTILINE)
        titles = [t.strip() for t in titles]
        parts = re.split(title_ptn, content, maxsplit=len(titles)+1, flags=re.MULTILINE)
        self.errors(content)
        index = titles.index("MODEL RESULTS")
        self.model_results_str = parts[index+1]
        if self.TITLES[1] in titles:
            index = titles.index(self.TITLES[1])
            self.std_model_results_str = parts[index+1]
        else:
            self.std_model_results_str = ''
        if self.TITLES[2] not in titles:
            raise ValueError(f"Can not find {self.TITLES[2]} in {titles}")
        self.model_fit_information_str = parts[titles.index(self.TITLES[2])+1]
        self.parts = parts
        self.titles = titles
        self.content = content
        
    @cached_property
    def model_results(self):
        "parse MODEL RESULTS"
        return self._parse_model_results(self.model_results_str)
    
    @cached_property
    def model_results_df(self)->pd.DataFrame:
        info = self.model_results
        df = pd.DataFrame(info['rows'], columns=['name'] + info['cols'])
        df.index = pd.MultiIndex.from_tuples(info['rowheads'])
        df.index.names = ('group', 'level', 'part')
        for c in info['cols']:
            df[c] = df[c].astype(float)
        return df
    
    @cached_property
    def stdyx_model_results_df(self)->pd.DataFrame:
        df = None
        for info in self.std_model_results:
            if info['title'] == 'STDYX Standardization':
                df = pd.DataFrame(info['rows'], columns=['name'] + info['cols'])
                df.index = pd.MultiIndex.from_tuples(info['rowheads'])
                df.index.names = ('group', 'level', 'part')
                for c in info['cols']:
                    df[c] = df[c].astype(float)
        return df
    
    def model_results_filter(self, df: pd.DataFrame, oper='BY'):
        index = df.index.map(lambda x : x[2].endswith(f'.{oper}'))
        return df[index]

    @cached_property
    def structure_results_df(self)->pd.DataFrame:
        df = self.model_results_filter(self.model_results_df, 'ON')
        stddf = self.model_results_filter(self.stdyx_model_results_df, 'ON')
        df.reset_index(inplace=True)
        stddf.reset_index(inplace=True)
        for c in ('level', 'group'):
            if (df[c]=='').all():
                del df[c]
                del stddf[c]
        def genpath(row: pd.Series):
            dep = row['part']
            if dep.endswith('.ON'):
                dep = dep[:-3]
            return f'{dep}<-{row["name"]}'
        df['Path'] = df.apply(genpath, axis=1)
        stddf['Path'] = df.apply(genpath, axis=1)
        df['StdEstimate'] = stddf['Estimate']
        return df[['Path', 'Estimate', 'StdEstimate', 'S.E.', 'Est./S.E.', 'P-Value']]
        

    @cached_property
    def stdy_model_results_df(self)->pd.DataFrame:
        df = None
        for info in self.std_model_results:
            if info['title'] == 'STDY Standardization':
                df = pd.DataFrame(info['rows'], columns=['name'] + info['cols'])
                df.index = pd.MultiIndex.from_tuples(info['rowheads'])
                df.index.names = ('group', 'level', 'part')
                for c in info['cols']:
                    df[c] = df[c].astype(float)
        return df
    
    @cached_property
    def std_model_results_df(self)->pd.DataFrame:
        df = None
        for info in self.std_model_results:
            if info['title'] == 'STD Standardization':
                df = pd.DataFrame(info['rows'], columns=['name'] + info['cols'])
                df.index = pd.MultiIndex.from_tuples(info['rowheads'])
                df.index.name = ('group', 'level', 'part')
                for c in info['cols']:
                    df[c] = df[c].astype(float)
        return df
    
    @cached_property
    def std_model_results(self)->List[dict]:
        "parse STANDARDIZED MODEL RESULTS"
        if not self.std_model_results_str:
            return
        Standardization_ptn = r"^STD.+Standardization$"
        titles = re.findall(Standardization_ptn, self.std_model_results_str, re.MULTILINE)
        contents = re.split(Standardization_ptn, self.std_model_results_str, maxsplit=len(titles)+1, flags=re.MULTILINE)
        results = []
        for i in range(len(titles)):
            r = self._parse_model_results(contents[i+1])
            r['title'] = titles[i]
            r['content'] = contents[i+1]
            results.append(r)
        return results
    
    @cached_property
    def model_fit_information(self)->list[tuple]:
        content = self.model_fit_information_str
        rows = []
        title = '' 
        for line in content.split('\n'):
            if not line or '(n* = (n + 2)' in line:
                continue
            pattern = r'-?\d+\.?\d+[*]?'  
            numbers = re.findall(pattern, line) 
            if numbers:
                num = numbers[0].strip('*')
                name = line.replace(num, ' ').strip()
                rows.append((title, name, num))
            else:
                if line[0] in '* ':
                    continue
                title = line.strip()
        lens = [len(r) for r in rows]
        assert max(lens) == min(lens)
        return rows
        
    @cached_property
    def model_fit_information_df(self)->pd.DataFrame:
        data = self.model_fit_information
        df = pd.DataFrame(
            data, 
            columns=('part', 'name', 'value')
        )
        df['value'] = df['value'].astype(float)
        return df

    @cached_property
    def factor_loadings(self)->pd.DataFrame:
        df = self.stdyx_model_results_df.copy(True)
        loadings = self.model_results_filter(df, 'BY')
        loadings.reset_index(inplace=True)
        if (loadings['level']=='').all():
            del loadings['level']
        if (loadings['group']=='').all():
            del loadings['group']
        cols = list(loadings.columns)
        index = cols.index('part')
        cols[index] = 'factor'
        loadings['part'] = loadings['part'].map(lambda x: x.split('.')[0])
        index = cols.index('name')
        cols[index] = 'item'
        loadings.columns = cols
        return loadings
    
    @cached_property
    def ave_cr(self)->pd.DataFrame:
        loadings = self.factor_loadings.copy(True)
        loadings['Estimate2'] = loadings['Estimate']**2
        loadings['epsilons'] = 1-loadings['Estimate2']
        loadings['n'] = 1
        cols = list(loadings.columns)
        i = cols.index('item')
        gdf = loadings.groupby(cols[:i])
        sm = gdf.sum()
        ave = sm['Estimate2']/sm['n']
        cr = sm['Estimate']**2/(sm['Estimate']**2 + sm['epsilons'])
        return pd.DataFrame({'AVE': ave, 'CR': cr})
    
    @cached_property
    def corr_matrix(self)->pd.DataFrame:
        params = self.stdyx_model_results_df.copy(deep=True)        
        params = self.model_results_filter(params, 'WITH')
        params.index = params.index.map(lambda x: x[2][:-5])
        cols = []
        index = []
        for v in params['name']:
            if v not in cols:
                cols.append(v)
        for v in params.index:
            if v not in cols:
                cols.append(v)
        df = pd.DataFrame(index=cols, columns=cols)
        for i in range(len(params)):
            idx = params.index[i]
            col = params['name'][i]
            df.loc[idx, col] = params['Estimate'].iloc[i]
        return df

    @cached_property
    def discriminant_df(self)->pd.DataFrame:
        '''区分效度表，表中元素是相关系数的平方，对角线元素是AVE'''
        crr = self.corr_matrix
        disdf = crr ** 2
        ave = self.ave_cr['AVE']
        for vn in disdf.index:
            disdf.loc[vn, vn] = ave[vn]
        return disdf
    
    @cached_property
    def indrect_df(self)->pd.DataFrame:
        title = self.TITLES[3]
        content = self.parts[self.titles.index(title) + 1]
        df = self._parse_indirect_results(content)
        return df
    
    def __parse_indirect_parts(self, partname, content: str)->list:
        numptn = re.compile(NUMBER_PTN)
        rowhead = []
        rows = []
        vname_ptn = re.compile(r'^\s+(' + VAR_NAME_PTN + ')$')
        for line in content.split('\n'):
            if not line.strip():continue
            numbers = numptn.findall(line)
            if 'Specific indirect' in line:
                continue
            elif len(numbers) == 4:
                eles = line.split()
                rowhead.append('.'.join(eles[:-4]))
                rows.append([partname, '->'.join(rowhead)] + eles[-4:])
                rowhead = []
            elif vname_ptn.match(line):
                rowhead.append(vname_ptn.findall(line)[0])
        return rows
                

    
    def _parse_indirect_results(self, content: str)->pd.DataFrame:
        cols = 'Path Effect Estimate       S.E.  Est./S.E.    P-Value'.split()
        part_ptn = re.compile(f'Effects from {VAR_NAME_PTN} to {VAR_NAME_PTN}')
        matches = part_ptn.findall(content)
        rows = []
        if matches:
            parts = part_ptn.split(content)
            for i, part  in enumerate(parts):
                if i == 0:continue
                rows += self.__parse_indirect_parts(matches[i-1], part)
        df = pd.DataFrame(rows, columns=cols)
        return df

    def _parse_model_results(self, content: str)->dict:
        MaxRowHeadLen = 10
        number_ptn = re.compile(r'\s([\-0-9\.]+)')
        cols = None
        rows = []
        rowHeads = []
        rowHead = []
        level = ''
        group = ''
        for line in content.split('\n'):
            data = line.strip().split(' ')
            data = [c for c in data if c]
            numbers = number_ptn.findall(line)
            if len(data)  == 0:
                continue
            if cols is None:
                if 'Estimate' in line: # col name line
                    cols = data
            elif line.lower() in ('between level', 'within level'):
                level = data[0]
            elif line.startswith("Group "):
                group = data[1]
            elif len(numbers) == 4: # data row
                rows.append(data)
                if rowHead:
                    rowHeads.append([level, group] + rowHead)
                    rowHead = []
                else:
                    rowHeads.append(rowHeads[-1])
            else: # row head
                rowHead.append('.'.join(data))
        assert len(rowHeads)==len(rows)
        if cols is None:
            raise ValueError("Can not find table column names in MODEL RESULTS")
        lens = [len(rh) for rh in rowHeads]
        assert max(lens) == min(lens), ValueError(f'rowHeads not the sampe length: {lens} {rowHeads}')
        return dict(
            cols=cols,
            rows=rows,
            rowheads=rowHeads,
        )

    def writeExcel(self, fpath: Path):
        # 验证 fpath 是有效的excel文件，然后将数据写入excel，一个dataframe写入一个worksheet
        fpath = Path(fpath)
        assert fpath.suffix in ('.xlsx', '.xls'), ValueError('fpath must have be a excel file')
        writer = pd.ExcelWriter(fpath, engine='xlsxwriter')
        for name in ('model_fit_information_df', 'factor_loadings', 'ave_cr', 'stdyx_model_results_df'):
            df = getattr(self, name)
            if df is not None:
                df.to_excel(writer, sheet_name=name)
        writer.close()

    def errors(self, content: str):
        ptn = re.compile(r'\*\*\* ERROR in [A-Z]+ command\n')
        ms = ptn.search(content)
        if ms:
            info = ptn.split(content)[1]
            err = ''
            for line in info.split('\n'):
                err += line
                if not line.strip():
                    break
            err += f'\n File is : {self.fpath}'
            raise MplusError(err)
        
    def _parse_single_column_data(self, content: str)->list:
        '''用于解析只有一列数据的字符串， 第一列是名称，第二列是数据
        例如：
            content = """
             VUONG-LO-MENDELL-RUBIN LIKELIHOOD RATIO TEST FOR 2 (H0) VERSUS 3 CLASSES

          H0 Loglikelihood Value                        -1180.844
          2 Times the Loglikelihood Difference            124.066
          Difference in the Number of Parameters               11
          Mean                                            -82.274
          Standard Deviation                              172.675
          P-Value                                          0.0156
        """
        '''
        rows = []
        for line in content.split('\n'):
            line = line.strip()
            parts = re.split(r'\s{2,}', line)
            if len(parts) == 2:
                rows.append(parts)
        return rows

        

def aggregate_fits(fitdfs: list[pd.DataFrame], titles: list[str]):
    '''
    多个模型的拟合指标聚合到一起，方便比较
    '''
    assert len(fitdfs) == len(titles), ValueError(f'{len(fitdfs)} != {len(titles)}')
    indices = [
        ('Chi-Square Test of Model Fit', 'Value', 'χ２'),
        ('Chi-Square Test of Model Fit', 'Degrees of Freedom', 'df'),
        ('CFI/TLI', 'CFI', 'CFI'),
        ('CFI/TLI', 'TLI', 'TLI'),
        ('RMSEA (Root Mean Square Error Of Approximation)', 'Estimate', 'RMSEA'),
        ('SRMR (Standardized Root Mean Square Residual)', 'Value', 'SRMR'),
    ]
    coldata = {}
    cols = []
    for part, name, col in indices:
        cols.append(col)
        if col not in coldata:
            coldata[col] = []
        for df in fitdfs:
            val = df[(df['part'] == part) & (df['name']==name)]['value'].iloc[0]
            coldata[col].append(val)
    fits = pd.DataFrame(coldata)
    fits['χ２/df'] = fits['χ２']/fits['df']
    cols.insert(2, 'χ２/df')
    fits = fits[cols]
    fits.index = titles
    return fits


