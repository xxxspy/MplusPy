from functools import cached_property
import pandas as pd
import re
from mpluspy import output
import os


class MplusModel:
    MaxLineWidth = 50
    CommondNames = ("TITLE", "DATA", "VARIABLE", "DEFINE",
              "MONTECARLO", "MODELPOPULATION", "MODELMISSING", "ANALYSIS",
              "MODEL", "MODELINDIRECT", "MODELCONSTRAINT", "MODELTEST", "MODELPRIORS",
              "OUTPUT", "SAVEDATA", "PLOT")
    CommondLabels = ("TITLE", "DATA", "VARIABLE", "DEFINE",
              "MONTECARLO", "MODEL POPULATION", "MODEL MISSING", "ANALYSIS",
              "MODEL", "MODEL INDIRECT", "MODEL CONSTRAINT", "MODEL TEST", "MODEL PRIORS",
              "OUTPUT", "SAVEDATA", "PLOT")

    def __init__(self,
                TITLE = None,
                DATA = None,
                VARIABLE = None,
                DEFINE = None,
                MONTECARLO = None,
                MODELPOPULATION = None,
                MODELMISSING = None,
                ANALYSIS = None,
                MODEL = None,
                MODELINDIRECT = [],
                MODELCONSTRAINT = None,
                MODELTEST = None,
                MODELPRIORS = None,
                OUTPUT = [],
                SAVEDATA = None,
                PLOT = None,
                usevariables = [],
                pdata: pd.DataFrame = None,
                autov = True,
                imputed = False,
                quiet = True,
                 ) -> None:
        self.TITLE = TITLE
        self._DATA = DATA
        self._VARIABLE: dict = VARIABLE or {}
        self.DEFINE = DEFINE
        self.MONTECARLO = MONTECARLO
        self.MODELPOPULATION = MODELPOPULATION
        self.MODELMISSING = MODELMISSING
        self.ANALYSIS = ANALYSIS
        self.MODEL = self.lines_shorter(MODEL)
        self._MODELINDIRECT = MODELINDIRECT
        self.MODELCONSTRAINT = MODELCONSTRAINT
        self.MODELTEST = MODELTEST
        self.MODELPRIORS = MODELPRIORS
        self._OUTPUT = OUTPUT
        self.SAVEDATA = SAVEDATA
        self.PLOT = PLOT
        self.usevariables = usevariables
        self.pdata = pdata
        self.autov = autov
        self.imputed = imputed
        self.quiet = quiet
        self.mplus_command = 'mplus'

    @cached_property
    def MODELINDIRECT(self)->str:
        return self._merge_(self._MODELINDIRECT)
    
    def _merge_(self, elements: list[str])->str:
        r = ''
        for o in elements:
            if o:
                if r:
                    r += '\n'
                r += o
                if not r.endswith(';'):
                    r += ';'
        return r

    @cached_property
    def OUTPUT(self)->str:
        if isinstance(self._OUTPUT, str):
            return self._OUTPUT
        r = ''
        for o in self._OUTPUT:
            if o:
                r += o
                if not r.endswith(';'):
                    r += ';'
        return r

    def lines_shorter(self, lines: str):
        newlines = []
        for line in lines.split('\n'):
            line = self.line_shorter(line)
            newlines.append(line)
        return '\n'.join(newlines)
        

    def line_shorter(self, line: str):
        if len(line) > self.MaxLineWidth:
            count = 0
            lines = []
            newline = ''
            for c in line:
                count += 1
                newline += c
                if count >= self.MaxLineWidth and c == ' ':
                    lines.append(newline)
                    count = 0
                    newline = ''
            if newline:
                lines.append(newline)
            line = '\n'.join(lines)
        return line


    @cached_property
    def VARIABLE(self)->str:
        varinfo = self._VARIABLE
        if 'missing' not in varinfo:
            varinfo['missing'] = '.'
        if 'names' not in varinfo:
            varinfo['names'] = self.line_shorter(' '.join(self.var_names))
        output = ''
        for k, v in varinfo.items():
            newline = ''
            if isinstance(v, str):
                newline = f'{k}={v};'
            elif isinstance(v, (list, tuple)):
                newline = f'{k}={" ".join(v)};'
            if newline:
                if output:
                    output += '\n    '
                output += newline
        return output
        

    @cached_property
    def syntax(self)->str:
        codes = []
        for name, label in zip(self.CommondNames, self.CommondLabels):
            content = getattr(self, name)
            if content:
                code = f'{label}:\n    {content}'
                if not code.endswith(';'):
                    code += ';'
                codes.append(code)
        return '\n'.join(codes)
    
    @cached_property
    def data_file(self)->str:
        return f'{self.TITLE}.dat'
    
    @cached_property
    def input_file(self)->str:
        return f'{self.TITLE}.inp'
    
    @cached_property
    def outpu_file(self)->str:
        return f'{self.TITLE}.out'
    
    @cached_property
    def DATA(self):
        if self._DATA:
            return self._DATA
        return f'FILE = "{self.data_file}"'
    
    def detect_vars_in_model(self)->list[str]:
        ptn = re.compile("^(.*)( by | BY | By | on | ON | On )(.*)$")
        vnames = []
        for line in self.MODEL.replace('\n', '').split(';'):
            mt = ptn.match(line)
            print(line, ptn.match(line))
            if mt:
                op = mt.group(2)
                vnames += line.replace(op, ' ').split(' ')
        cleaned = []
        for v in vnames:
            if v: 
                if v not in cleaned:
                    if self.pdata is not None:
                        if v in self.pdata.columns:
                            cleaned.append(v)
                    else:
                        cleaned.append(v)
        return cleaned

        
    def gen_data_file(self):
        df = self.pdata
        cols = self.var_names
        subdf = df[cols]
        subdf.to_csv(self.data_file, index=None, header=False, sep=' ')

    @cached_property
    def vnames_in_VARIABLE(self)->list[str]:
        vnames = []
        if self._VARIABLE:
            if 'names' in self._VARIABLE:
                vnames = self._VARIABLE['names']
            elif 'usevariables' in self._VARIABLE:
                vnames = self._VARIABLE['names']
        return vnames

    @cached_property
    def var_names(self)->list:
        if self.usevariables:
            cols = self.usevariables
        else:
            cols = self.vnames_in_VARIABLE or self.detect_vars_in_model()
        return cols

    def fit(self)->output.MplusParser:
        self.gen_data_file()
        with open(self.input_file, 'w', encoding='utf8') as f:
            f.write(self.syntax)
        os.system(f'{self.mplus_command} {self.input_file}')
        return output.MplusParser(self.outpu_file)
    

        