import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression
import scipy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

patient_IDs = pd.read_csv('PT REPORT Pendulum.csv')['Recent KIT ID']
pop_data = pd.read_csv('PopulationData_GladdenLongevity.csv').set_index('Patient.ID')

patient_IDs.replace('No kit', np.nan, inplace=True)
patient_IDs.dropna(inplace=True)
print(patient_IDs)
treated_indices = pop_data[[x in (set(patient_IDs) & set(pop_data['Kit ID'])) for x in pop_data['Kit ID']]].index
print(treated_indices)
print(pop_data)

age_differentials = pd.DataFrame(columns=['Chronological Age',
                                          'GrimAgePC',
                                          'PhenoAgePC',
                                          'HorvathPC',
                                          'HannumPC',
                                          'TelomerePC',
                                          'DunedinPACE',
                                          'Immune.CD8T',
                                          'Immune.CD4T',
                                          'Immune.CD4T.CD8T',
                                          'Immune.NK',
                                          'Immune.Bcell',
                                          'Immune.Mono',
                                          'Immune.Neutrophil',
                                          # 'AltumAge'
                                          ])
for patient in pop_data.index:
    chrono_age = float(pop_data.loc[patient, 'Decimal Chronological Age'])
    grim_r = float(pop_data.loc[patient, 'GrimAge PC ']) - chrono_age
    pheno_r = float(pop_data.loc[patient, 'PhenoAge PC']) - chrono_age
    horPC = float(pop_data.loc[patient, 'Horvath PC']) - chrono_age
    hanPC = float(pop_data.loc[patient, 'Hannum PC']) - chrono_age
    teloPC = float(pop_data.loc[patient, 'Telomere Values'])
    dune = float(pop_data.loc[patient, 'DunedinPoAm'])
    # altum = float(pop_data.loc[patient, 'AltumAge']) - chrono_age
    cd8 = float(pop_data.loc[patient, 'Immune.CD8T'])
    cd4 = float(pop_data.loc[patient, 'Immune.CD4T'])
    cdRatio = float(pop_data.loc[patient, 'Immune.CD4T.CD8T'])
    nk = float(pop_data.loc[patient, 'Immune.NK'])
    bCell = float(pop_data.loc[patient, 'Immune.Bcell'])
    mono = float(pop_data.loc[patient, 'Immune.Mono'])
    neutro = float(pop_data.loc[patient, 'Immune.Neutrophil'])

    age_differentials.loc[patient] = chrono_age, \
                                     grim_r, \
                                     pheno_r,\
                                     horPC, \
                                     hanPC, \
                                     teloPC, \
                                     dune, \
                                     cd8, \
                                     cd4, \
                                     cdRatio, \
                                     nk, \
                                     bCell, \
                                     mono, \
                                     neutro

treated = age_differentials.loc[treated_indices]
untreated = age_differentials.drop(index=treated_indices)

tests = pd.DataFrame(columns=['Kruskal-Wallis', 'KW p-Value', 'T-test', ' TT p-Value', 'Average Change'])
for clock in age_differentials.columns[1:]:
    kruskal, k_p = stats.kruskal(treated[clock], untreated[clock])
    ttest, t_p = stats.ttest_ind(treated[clock], untreated[clock])
    unt_avg, treat_avg = np.average(untreated[clock]), np.average(treated[clock])
    avg_change = treat_avg - unt_avg
    tests.loc[clock] = [kruskal, k_p, ttest, t_p, avg_change]

    fig = plt.figure()
    box = fig.add_subplot()
    box.boxplot([untreated[clock], treated[clock]])
    box.set_xticklabels(['Untreated', 'Treated'])
    box.set_title(clock + '\nTT p-Value=' + str(round(t_p, 4)) + ', KW p-Value=' + str(round(k_p, 4)) +
                  ', Untreated n=' + str(len(untreated[clock])) + ', Treated n=' + str(len(treated[clock])))
    plt.savefig('Pendulum Figures/' + clock + '.png')
    plt.show()

tests.to_csv('PendulumAnalysis.csv')


