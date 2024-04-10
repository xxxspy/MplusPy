from pathlib import Path
import re
from typing import List
import pandas as pd
from functools import cached_property

class MplusParser:
    TITLES = ("MODEL RESULTS", "STANDARDIZED MODEL RESULTS", "MODEL FIT INFORMATION")
    
    def __init__(self, fpath: str) -> None:
        self.fpath = fpath
        content = Path(fpath).read_text()
        title_ptn = r"^\n[A-Z][A-Z \-]*\n$"
        titles = re.findall(title_ptn, content, re.MULTILINE)
        titles = [t.strip() for t in titles]
        parts = re.split(title_ptn, content, maxsplit=len(titles)+1, flags=re.MULTILINE)
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
        df.reset_index(inplace=True)
        if (df['level']=='').all():
            del df['level']
        if (df['group']=='').all():
            del df['group']
        
        loadings = df[df['part'].map(lambda x : '.BY' in x)]
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


    def _parse_model_results(self, content: str):
        cols = None
        rows = []
        rowHeads = []
        rowHead = []
        level = ''
        group = ''
        for line in content.split('\n'):
            data = line.strip().split(' ')
            data = [c for c in data if c]
            if len(data)  == 0:
                continue
            if cols is None:
                if 'Estimate' in line: # col name line
                    cols = data
            elif line.lower() in ('between level', 'within level'):
                level = data[0]
            elif line.startswith("Group "):
                group = data[1]
            elif len(data) == 5:
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
        assert max(lens) == min(lens), ValueError(rowHeads)
        return dict(
            cols=cols,
            rows=rows,
            rowheads=rowHeads,
        )
