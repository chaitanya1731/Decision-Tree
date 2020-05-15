import math
from collections import Counter

class Entropy(object):
    @classmethod
    def entropy(self, probs):
        return sum([-prob * math.log(prob, 2) for prob in probs])

    @classmethod
    def entropy_of_list(self, a_list):
        cnt = Counter(x for x in a_list)
        num_instances = len(a_list) * 1.0
        probs = [x / num_instances for x in cnt.values()]
        return self.entropy(probs)

    @classmethod
    def information_gain(self, df, split_attribute_name, target_attribute_name, trace=0):
        df_split = df.groupby(split_attribute_name)
        nobs = len(df.index) * 1.0
        df_agg_ent = df_split.agg({target_attribute_name: [self.entropy_of_list, lambda x: len(x) / nobs]})[
            target_attribute_name]
        df_agg_ent.columns = ['Entropy', 'PropObservations']
        newEntropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
        oldEntropy = self.entropy_of_list(df[target_attribute_name])
        return oldEntropy - newEntropy