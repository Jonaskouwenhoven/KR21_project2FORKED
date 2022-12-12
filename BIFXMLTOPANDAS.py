import xml.etree.ElementTree as ET
import pandas as pd
import itertools
# parse the BIFXML file and get the root element
tree = ET.parse('/Users/jonas/Documents/GitHub/KR21_project2FORKED/KR21_forked/testing/Russia.BIFXML')
root = tree.getroot()

# find all the CPTS in the file
cpts = root.findall('.//DEFINITION')

# create a list to store the dataframes
dataframes = []

# iterate over the CPTS and convert them to Pandas dataframes
for cpt in cpts:
    # get the CPT name
    try:
        name = cpt.find('.//FOR').text
    except:
        continue
    # print(name)
    # create a dataframe with the CPT name as the index
    df = pd.DataFrame(index=[name])
    df = pd.DataFrame(columns=['outcome', 'probability'], index=[name])
    # get the probabilities from the TABLE element
    try:
        probabilities = cpt.find('.//TABLE').text.split()
    except:
        continue
    try:
        givens = cpt.find('.//GIVEN').text.split()
    except:
        continue
    # print(givens)
    # create a list of outcomes for the CPT 
    outcomes = [
        outcome.text
        for outcome in cpt.findall('.//GIVEN')
    ]
    outcomes.append(name)
    outcomes.append('probability')

    ding = []
    combinations = list(itertools.product([False, True], repeat=3))
    for i, comb in enumerate(combinations):
        temp = [c for c in comb]
        temp.append(probabilities[i])
        ding.append(temp)
        
    df = pd.DataFrame(ding, columns=outcomes)
    dataframes.append(df)

for df in dataframes:
    print(df.to_latex(index=False))
    print("\n")
