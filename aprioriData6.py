import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('./data/data6.csv', sep=';')  

print(df.head())

print("\nLinhas brutas do arquivo:")
with open('./data/data6.csv', 'r', encoding='latin1') as f:
    for _ in range(5):
        print(f.readline())

transactions = [
    ['leite', 'manteiga', 'pão'],
    ['pão', 'manteiga'],
    ['leite', 'pão'],
    ['leite', 'manteiga'],
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, num_itemsets=1, min_threshold=0.5)

print("Regras de Associação com Suporte 50% e Confiança 50%")
print(rules)

frequent_itemsets2 = apriori(df, min_support=0.5, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, num_itemsets=1, min_threshold=0.75)

print("Regras de Associação com Suporte 50% e Confiança 75%")
print(rules2)
