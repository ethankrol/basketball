from fuzzywuzzy import fuzz
name = 'N.C. State'
name_2 = 'MI State'

print(f'Similarity score: {fuzz.token_set_ratio(name, name_2)}')