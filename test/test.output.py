import unittest
from pathlib import Path
from mpluspy.output import MplusParser
from typing import List

DATADIR = Path(__file__).absolute().parent.parent / 'testdata'

class TestStringMethods(unittest.TestCase):
    mps: List[MplusParser]
    def setUp(self):
        self.fpaths = ( 
            DATADIR/'mplus/cfadata.out',    
            DATADIR/'mplus/9a.out',   
            DATADIR/'mplus/ex5.14.out')
        self.mps = []
        for fp in self.fpaths:
            mp = MplusParser(fp)
            self.mps.append(mp)

    def test_model_fits(self):
        mf = self.mps[0].model_fit_information
        self.assertEqual(len(mf), 21)
        self.assertEqual(mf[0][2], '87')
        self.assertEqual(mf[0][1], 'Number of Free Parameters')
        self.assertEqual(mf[1][0], 'Loglikelihood')
        df = self.mps[0].model_fit_information_df

    def test_model_results(self):
        mf = self.mps[2].model_results
        self.assertListEqual(mf['cols'], ['Estimate', 'S.E.', 'Est./S.E.', 'P-Value'])
        self.assertListEqual(mf['rowheads'], [['', 'MALE', 'F1.BY'], ['', 'MALE', 'F1.BY'], ['', 'MALE', 'F1.BY'], ['', 'MALE', 'F2.BY'], ['', 'MALE', 'F2.BY'], ['', 'MALE', 'F2.BY'], ['', 'MALE', 'F1.ON'], ['', 'MALE', 'F1.ON'], ['', 'MALE', 'F1.ON'], ['', 'MALE', 'F2.ON'], ['', 'MALE', 'F2.ON'], ['', 'MALE', 'F2.ON'], ['', 'MALE', 'F2.WITH'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'MALE', 'Residual.Variances'], ['', 'FEMALE', 'F1.BY'], ['', 'FEMALE', 'F1.BY'], ['', 'FEMALE', 'F1.BY'], ['', 'FEMALE', 'F2.BY'], ['', 'FEMALE', 'F2.BY'], ['', 'FEMALE', 'F2.BY'], ['', 'FEMALE', 'F1.ON'], ['', 'FEMALE', 'F1.ON'], ['', 'FEMALE', 'F1.ON'], ['', 'FEMALE', 'F2.ON'], ['', 'FEMALE', 'F2.ON'], ['', 'FEMALE', 'F2.ON'], ['', 'FEMALE', 'F2.WITH'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances'], ['', 'FEMALE', 'Residual.Variances']])
        self.assertEqual(len(mf['rows']), len(mf['rowheads']))
        self.assertEqual(len(mf['cols']), len(mf['rows'][0])-1)
        df = self.mps[2].model_results_df

    def test_std_model_results(self):
        mf = self.mps[1].model_results
        self.assertListEqual(mf['cols'], ['Estimate', 'S.E.', 'Est./S.E.', 'P-Value'])
        self.assertListEqual(mf['rowheads'], [['Within', '', 'Y.ON'], ['Within', '', 'Residual.Variances'], ['Between', '', 'Y.ON'], ['Between', '', 'Y.ON'], ['Between', '', 'Intercepts'], ['Between', '', 'Residual.Variances']])
        self.assertEqual(len(mf['rows']), len(mf['rowheads']))
        self.assertEqual(len(mf['cols']), len(mf['rows'][0])-1)
        df1 = self.mps[0].stdyx_model_results_df
        df2 = self.mps[0].stdy_model_results_df
        df3 = self.mps[0].std_model_results_df

    def test_cfa(self):
        fl = self.mps[0].factor_loadings
        avecr = self.mps[0].ave_cr

    def test_write_excel(self):
        self.mps[0].writeExcel('d:/text.xlsx')




if __name__ == '__main__':

    unittest.main()