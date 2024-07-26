from .output import MplusParser
from functools import cached_property
import re
import pandas as pd

class LCAParser(MplusParser):
    
    @cached_property
    def fit_indices(self)->dict:
        fits = self.model_fit_information
        def find(name1: str, name2: str)->str:
            for fit in fits:
                if name1 == fit[0] and name2 == fit[1]:
                    return fit[2]
            raise ValueError('Fit not found', name1, name2)
        indices = {
            'AIC': find('Information Criteria', 'Akaike (AIC)'),
            'BIC': find('Information Criteria', 'Bayesian (BIC)'),
            'aBIC': find('Information Criteria', 'Sample-Size Adjusted BIC'),
        }
        return indices

    @cached_property
    def entropy(self)->str:
        if 'CLASSIFICATION QUALITY' not in self.titles: return '-'
        part = self.parts[self.titles.index('CLASSIFICATION QUALITY')+1]
        ptn = re.compile(r'Entropy\s+(\d+\.\d+)')
        finds = ptn.search(part)
        if finds is None:
            raise ValueError('Entropy not found')
        return finds.group(1)
    
    @cached_property
    def class_percents(self)->pd.DataFrame:
        percents = []
        index0 = self.content.index('BASED ON THE ESTIMATED MODEL')
        index1 = self.content.index('\n\n\n', index0)
        data = self.content[index0:index1].split('\n')
        number_ptn = re.compile(r'[\d \.]+')
        for line in data:
            line = line.strip()
            if not line: continue
            if number_ptn.match(line):
                percents.append(line.split())
        
        df = pd.DataFrame(percents)
        df.columns = ['class', 'frequency', 'percent']
        return df

    @cached_property
    def lmr_test(self)->pd.DataFrame:
        names = [
            "H0 Loglikelihood Value                ",
            "2 Times the Loglikelihood Difference  ",
            "Difference in the Number of Parameters",
            "Mean                                  ",
            "Standard Deviation                    ",
            "P-Value                               ",
        ]
        names = [name.strip() for name in names]
        if 'TECHNICAL 11 OUTPUT' not in self.titles: 
            return pd.DataFrame({
                'name': names,
                'value': '-',

            })
        content = self.parts[self.titles.index('TECHNICAL 11 OUTPUT')+1]
        rows = self._parse_single_column_data(content)

        rows = [row for row in rows if row[0] in names]
        df = pd.DataFrame(rows)
        df.columns = ['name', 'value']
        df.iloc[-1, 0] = 'adj P-Value'
        return df
    
    @cached_property
    def blrt_test(self)->pd.DataFrame:
        names = [
          "H0 Loglikelihood Value                        ",
          "2 Times the Loglikelihood Difference          ",
          "Difference in the Number of Parameters        ",
          "Approximate P-Value                           ",
          "Successful Bootstrap Draws                    ",
        ]
        names = [name.strip() for name in names]
        if 'TECHNICAL 14 OUTPUT' not in self.titles: 
            return pd.DataFrame({
                'name': names,
                'value': '-',

            })
        content = self.parts[self.titles.index('TECHNICAL 14 OUTPUT')+1]
        rows = self._parse_single_column_data(content)
        names = [
          "H0 Loglikelihood Value                        ",
          "2 Times the Loglikelihood Difference          ",
          "Difference in the Number of Parameters        ",
          "Approximate P-Value                           ",
          "Successful Bootstrap Draws                    ",
        ]
        names = [name.strip() for name in names]
        rows = [row for row in rows if row[0] in names]
        df = pd.DataFrame(rows)
        df.columns = ['name', 'value']
        return df
    
    @cached_property
    def item_class_probability(self)->pd.DataFrame:
        content = self.parts[self.titles.index('RESULTS IN PROBABILITY SCALE')+1]
        parts = re.split(r'\s+Latent Class \d\n', content)[1:]
        def parse_part(part: str):
            itemname = ''
            for line in part.split('\n'):
                line = line.strip()
                if not line: continue
                if ' ' not in line:
                    itemname = line
                else:
                    row = line.split()
                    row.insert(0, itemname)
                    yield row
        data = []
        for i in range(len(parts)):
            part = parts[i]
            for row in parse_part(part):
                row.insert(0, i+1)
                data.append(row)
        df = pd.DataFrame(data)
        df.columns = ['class', 'item', '-', 'CategoryOrder', 'Probability', 'SE', 'Est/SE', 'P']
        return df[['class', 'item', 'CategoryOrder', 'Probability',]]
                
            


def aggregate_fits(filepaths: list, modelnames: list=[])->pd.DataFrame:
    parsers = [LCAParser(fpath) for fpath in filepaths]
    data = []
    for p in parsers:
        cp = p.class_percents
        if cp.shape[0] == 1:
            percents = '-'
        else:
            ps = []
            for per in cp['percent']:
                ps.append(per)
            percents = '/'.join([str(p) for p in ps])
        data.append({
            'AIC': p.fit_indices['AIC'],
            'BIC': p.fit_indices['BIC'],
            'aBIC': p.fit_indices['aBIC'],
            'Entropy': p.entropy,
            'P-LMR': p.lmr_test.loc[p.lmr_test['name']=='P-Value', 'value'].iloc[0],
            'P-BLRT': p.blrt_test.loc[p.blrt_test['name']=='Approximate P-Value', 'value'].iloc[0],
            'Percents': percents,
        })
    df = pd.DataFrame(data)
    cols = ['AIC', 'BIC', 'aBIC', 'Entropy', 'P-LMR', 'P-BLRT', 'Percents']
    df.columns = cols
    if modelnames:
        df['Model'] = modelnames
        cols.insert(0, 'Model')
    return df[cols]
