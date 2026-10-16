with open('F:/mp-latest/mp_proj_1/floodwatch_ieee.tex', 'r') as f:
    content = f.read()

checks = [
    ('GRU test split info', 'N=70 test samples, 8 positive'),
    ('GRU AUC', '0.8952'),
    ('GRU Brier', '0.0469'),
    ('M7 formula', 'R=100\\,H'),
    ('Trust gate', 'candidate.trust_flag'),
    ('M5 ensemble', 'same base checkpoint'),
    ('M5 perturbation', 'noise'),
    ('Alert policy value', 'predicted_flood_probability'),
    ('Logistic regression baseline', 'logistic regression'),
    ('Logistic AUC', '0.9879'),
]

for name, text in checks:
    found = text in content
    print(f'{name}: {"FOUND" if found else "MISSING"}')