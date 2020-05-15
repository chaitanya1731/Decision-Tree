import numpy as np
class Variance(object):
    def unique(seq, return_counts=False, id=None):
        found = set()
        if id is None:
            for x in seq:
                found.add(x)
        else:
            for x in seq:
                x = id(x)
                if x not in found:
                    found.add(x)
        found = list(found)
        counts = [seq.count(0), seq.count(1)]
        if return_counts:
            return found, counts
        else:
            return found

    @classmethod
    def calculate_variance(self, target_values):
        values = list(target_values)
        elements, counts = self.unique(values, True)
        variance_impurity = 0
        sum_counts = np.sum(counts)
        for i in elements:
            variance_impurity += (-counts[i] / sum_counts * (counts[i] / sum_counts))
        return variance_impurity

    @classmethod
    def variance_gain(self, df, split_attribute_name, target_attribute_name, trace=0):
        # Split Data by Possible Vals of Attribute:
        splitDataset = df.groupby(split_attribute_name)
        nobs = len(df.index) * 1.0
        df_agg_ent = splitDataset.agg({target_attribute_name: [self.calculate_variance, lambda x: len(x) / nobs]})[target_attribute_name]
        df_agg_ent.columns = ['Variance', 'VarObservation']
        newVariance = sum(df_agg_ent['Variance'] * df_agg_ent['VarObservation'])
        oldVariance = self.calculate_variance(df[target_attribute_name])
        return oldVariance - newVariance
