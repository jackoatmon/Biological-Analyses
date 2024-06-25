import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression
import scipy
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')

run_continuous = False
run_categorical = False
save_figs = True
show_continuous = False
show_demographics = False
show_residuals = False
best_fit = True
new_uniques = False


pendulum = pd.read_csv('PopulationData_GladdenLongevity.csv').set_index('Patient.ID')
kits = pd.read_csv('PT REPORT Pendulum.csv')['Recent KIT ID']
bad_indices = kits[kits == 'No kit'].index
kits.drop(index=bad_indices, inplace=True)

totalOnes = 0
good_kits = []
# for patient in pendulum.index:
#     if pendulum.loc[patient, 'Kit ID'] in kits.values:
#         pendulum.loc[patient, 'Treated'] = 1
#         totalOnes += 1
#         good_kits.append(pendulum.loc[patient, 'Kit ID'])
#     else:
#         pendulum.loc[patient, 'Treated'] = 0
#         print(patient, pendulum.loc[patient, 'Kit ID'])
#
# for kit in kits:
#     if kit not in good_kits:
#         print(kit)
# print(totalOnes)
# pendulum.to_csv('PopulationData_GladdenLongevity.csv')

covariates = pd.read_csv('Jack_PatientMetaData_052522 - Organized.csv').set_index('PatientID')
clock_data = pd.read_csv('PopulationData_060822.csv').set_index('Patient ID')

for col in covariates.columns:
    try:
        covariates[col] = covariates[col].fillna('nan').apply(str.lower).replace('nan', np.nan)
    except:
        pass

age_differentials = pd.DataFrame(columns=['Chronological Age',
                                          'GrimAgePC',
                                          'PhenoAgePC',
                                          'HorvathPC',
                                          'HannumPC',
                                          'TelomerePC',
                                          'DunedinPACE',
                                          'Full_CumulativeStemCellDivisions',
                                          'Avg_LifeTime_IntrinsicStemCellDivisionRate',
                                          'Median_Lifetime_IntrinsicStemCellDivisionRate',
                                          'Immune.CD8T',
                                          'Immune.CD4T',
                                          'Immune.CD4T.CD8T',
                                          'Immune.NK',
                                          'Immune.Bcell',
                                          'Immune.Mono',
                                          'Immune.Neutrophil',
                                          # 'AltumAge'
                                          ])
immunes = ['Full_CumulativeStemCellDivisions',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate',
           'Median_Lifetime_IntrinsicStemCellDivisionRate',
           'Immune.CD8T',
           'Immune.CD4T',
           'Immune.CD4T.CD8T',
           'Immune.NK',
           'Immune.Bcell',
           'Immune.Mono',
           'Immune.Neutrophil']

shared_patients = list(set(covariates.index) & set(clock_data.index))

print(clock_data['Decimal Chronological Age'])

drop_indices = list(clock_data.loc[clock_data['Decimal Chronological Age'].isnull()].index)
clocks = ['GrimAge PC ',
          'PhenoAge PC',
          'Horvath PC',
          'Hannum PC']
for clock in clocks:
    drop_indices = drop_indices + list(clock_data.loc[clock_data[clock].isnull()].index)
grimA, grimB = np.polyfit(clock_data['Decimal Chronological Age'].drop(index=drop_indices), clock_data['GrimAge PC '].drop(index=drop_indices), 1)
phenoA, phenoB = np.polyfit(clock_data['Decimal Chronological Age'].drop(index=drop_indices), clock_data['PhenoAge PC'].drop(index=drop_indices), 1)
horA, horB = np.polyfit(clock_data['Decimal Chronological Age'].drop(index=drop_indices), clock_data['Horvath PC'].drop(index=drop_indices), 1)
hanA, hanB = np.polyfit(clock_data['Decimal Chronological Age'].drop(index=drop_indices), clock_data['Hannum PC'].drop(index=drop_indices), 1)

print(grimA, grimB)
for patient in shared_patients:
    chrono_age = float(clock_data.loc[patient, 'Decimal Chronological Age'])

    if best_fit:
        grimVal = float(clock_data.loc[patient, 'GrimAge PC '])
        phenoVal = float(clock_data.loc[patient, 'PhenoAge PC'])
        horVal = float(clock_data.loc[patient, 'Horvath PC'])
        hanVal = float(clock_data.loc[patient, 'Hannum PC'])
        best_fit_grim = grimA * chrono_age + grimB
        best_fit_pheno = phenoA * chrono_age + phenoB
        best_fit_hor = horA * chrono_age + horB
        best_fit_han = hanA * chrono_age + hanB

        grim_r = grimVal - best_fit_grim
        pheno_r = phenoVal - best_fit_pheno
        horPC = horVal - best_fit_hor
        hanPC = hanVal - best_fit_han
    else:
        grim_r = float(clock_data.loc[patient, 'GrimAge PC ']) - chrono_age
        pheno_r = float(clock_data.loc[patient, 'PhenoAge PC']) - chrono_age
        horPC = float(clock_data.loc[patient, 'Horvath PC']) - chrono_age
        hanPC = float(clock_data.loc[patient, 'Hannum PC']) - chrono_age
    teloPC = float(clock_data.loc[patient, 'Telomere Values'])
    dune = float(clock_data.loc[patient, 'DunedinPoAm'])
    # altum = float(clock_data.loc[patient, 'AltumAge']) - chrono_age
    fullCell = float(clock_data.loc[patient, 'Full_CumulativeStemCellDivisions'])
    avgCell = float(clock_data.loc[patient, 'Avg_LifeTime_IntrinsicStemCellDivisionRate'])
    medianCell = float(clock_data.loc[patient, 'Median_Lifetime_IntrinsicStemCellDivisionRate'])
    cd8 = float(clock_data.loc[patient, 'Immune.CD8T'])
    cd4 = float(clock_data.loc[patient, 'Immune.CD4T'])
    cdRatio = float(clock_data.loc[patient, 'Immune.CD4T.CD8T'])
    nk = float(clock_data.loc[patient, 'Immune.NK'])
    bCell = float(clock_data.loc[patient, 'Immune.Bcell'])
    mono = float(clock_data.loc[patient, 'Immune.Mono'])
    neutro = float(clock_data.loc[patient, 'Immune.Neutrophil'])

    age_differentials.loc[patient] = chrono_age, \
                                     grim_r, \
                                     pheno_r,\
                                     horPC, \
                                     hanPC, \
                                     teloPC, \
                                     dune, \
                                     fullCell, \
                                     avgCell, \
                                     medianCell, \
                                     cd8, \
                                     cd4, \
                                     cdRatio, \
                                     nk, \
                                     bCell, \
                                     mono, \
                                     neutro


age_differentials.to_csv('PatientClockDifferentials.csv')

n_list = ['Cardiovascular',
          'Respiratory Disease',
          'Endocrine Disease',
          'Gastrointestinal',
          'Genito-Urinary',
          'Musculoskeletal',
          'Neuropsychological',
          'Reproductive',
          'Immune',
          'Do you take any of the Following nutritional Supplements?',
          'Do you take any of the following supplements or medications?',
          'Recreational Drug Use?',
          ]
for nl in n_list:
    covariates[nl].replace(np.nan, 'none', inplace=True)

differentials = pd.read_csv('PatientClockDifferentials.csv').set_index('Unnamed: 0').dropna()
differentials = differentials.astype(float)

continuous = ['Weight (lb)',
              'total height (in)',
              'BMI',
              'Pack Years (If smoker)',
              'Stress Level',
              'Age for Loss of Hair',
              'Total Disease Count',
              'DC Cardiovascular',
              'DC Respiratory Disease',
              'DC Endocrine Disease',
              'DC Gastrointestinal',
              'DC Genito-Urinary',
              'DC Musculoskeletal',
              'DC Neuropsychological',
              'DC Reproductive',
              'DC Immune',
              ]
categorical = ['Biological Sex',
               'Blood Type',
               'Alcohol Use(times per week)',
               'Menopause',
               'Given Birth',
               'Mother Nicotine Use',
               'Mother Pregnancy Complications',
               'Diagnosed with low or high calcium',
               'Marital Status',
               'Caffeine Use',
               'Tobacco Use',
               'How often do you use recreational Drugs?',
               'How often do you exercise per week?',
               'Exercise Type?',
               'Sexual Frequency?',
               'Hours of sleep per night?',
               'Level of Education',
               'Education of Mother',
               'Education of Father',
               'Ethnicity',
               'Cardiovascular',
               'Respiratory Disease',
               'Endocrine Disease',
               'Gastrointestinal',
               'Genito-Urinary',
               'Musculoskeletal',
               'Neuropsychological',
               'Reproductive',
               'Immune',
               # 'Total Disease Count',
               # 'DC Cardiovascular',
               # 'DC Respiratory Disease',
               # 'DC Endocrine Disease',
               # 'DC Gastrointestinal',
               # 'DC Genito-Urinary',
               # 'DC Musculoskeletal',
               # 'DC Neuropsychological',
               # 'DC Reproductive',
               # 'DC Immune',
               'Do you take any of the Following nutritional Supplements?',
               'Do you take any of the following supplements or medications?',
               'Recreational Drug Use?',
               'What does your diet mostly consist of?',
               'Arthritis (FH)',
               'Asthma (FH)',
               'Back Problems (FH)',
               'Blood Diseases (FH)',
               'Cancer (FH)',
               'COPD (FH)',
               'Diabetes (FH)',
               'Drug/Alcohol (FH)',
               'Emphysema (FH)',
               'Genetic Disorder (FH)',
               'Stomach Problems (FH)',
               'Kidney Disease (FH)',
               'Heart Problems (FH)',
               'High Blood Pressure (FH)',
               'Neurological Disease (FH)',
               'Obesity (FH)',
               'Psychiatric (FH)',
               'Scoliosis (FH)',
               'SIDS (FH)',
               'Stroke (FH)',
               'Tuberculosis (FH)',
               'Thyroid Disorder (FH)'
               ]
ordinals = ['Alcohol Use(times per week)',
            'Caffeine Use',
            'Tobacco Use',
            'How often do you use recreational Drugs?',
            'How often do you exercise per week?',
            'Sexual Frequency?',
            'Hours of sleep per night?',
            'Level of Education',
            'Education of Mother',
            'Education of Father',
            'Menopause'
            ]
for o in range(len(ordinals)):
    ordinals[o] = ordinals[o].lower()

'''NOTES
** get demographics on pace of aging for each category
** box and whisker for each of the possible responses (of their residuals/aging rates)

'''

og_cols = ['GrimAgePC Pearson',
           'GrimAgePC Pearson p-Value',
           'GrimAgePC Spearman',
           'GrimAgePC Spearman p-Value',
           'GrimAgePC Entropy/Importance Continuous',

           'PhenoAgePC Pearson',
           'PhenoAgePC Pearson p-Value',
           'PhenoAgePC Spearman',
           'PhenoAgePC Spearman p-Value',
           'PhenoAgePC Entropy/Importance Continuous',

           'HorvathPC Pearson',
           'HorvathPC Pearson p-Value',
           'HorvathPC Spearman',
           'HorvathPC Spearman p-Value',
           'HorvathPC Entropy/Importance Continuous',

           'HannumPC Pearson',
           'HannumPC Pearson p-Value',
           'HannumPC Spearman',
           'HannumPC Spearman p-Value',
           'HannumPC Entropy/Importance Continuous',

           'TelomerePC Pearson',
           'TelomerePC Pearson p-Value',
           'TelomerePC Spearman',
           'TelomerePC Spearman p-Value',
           'TelomerePC Entropy/Importance Continuous',

           'DunedinPACE Pearson',
           'DunedinPACE Pearson p-Value',
           'DunedinPACE Spearman',
           'DunedinPACE Spearman p-Value',
           'DunedinPACE Entropy/Importance Continuous',

           # 'AltumAge Pearson',
           # 'AltumAge Pearson p-Value',
           # 'AltumAge Spearman',
           # 'AltumAge Spearman p-Value',
           # 'AltumAge Entropy/Importance Continuous',

           'Full_CumulativeStemCellDivisions Pearson',
           'Full_CumulativeStemCellDivisions Pearson p-Value',
           'Full_CumulativeStemCellDivisions Spearman',
           'Full_CumulativeStemCellDivisions Spearman p-Value',
           'Full_CumulativeStemCellDivisions Entropy/Importance Continuous',

           'Avg_LifeTime_IntrinsicStemCellDivisionRate Pearson',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Pearson p-Value',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Spearman',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Spearman p-Value',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Entropy/Importance Continuous',

           'Median_Lifetime_IntrinsicStemCellDivisionRate Pearson',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Pearson p-Value',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Spearman',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Spearman p-Value',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Entropy/Importance Continuous',

           'Immune.CD8T Pearson',
           'Immune.CD8T Pearson p-Value',
           'Immune.CD8T Spearman',
           'Immune.CD8T Spearman p-Value',
           'Immune.CD8T Entropy/Importance Continuous',

           'Immune.CD4T Pearson',
           'Immune.CD4T Pearson p-Value',
           'Immune.CD4T Spearman',
           'Immune.CD4T Spearman p-Value',
           'Immune.CD4T Entropy/Importance Continuous',

           'Immune.CD4T.CD8T Pearson',
           'Immune.CD4T.CD8T Pearson p-Value',
           'Immune.CD4T.CD8T Spearman',
           'Immune.CD4T.CD8T Spearman p-Value',
           'Immune.CD4T.CD8T Entropy/Importance Continuous',

           'Immune.NK Pearson',
           'Immune.NK Pearson p-Value',
           'Immune.NK Spearman',
           'Immune.NK Spearman p-Value',
           'Immune.NK Entropy/Importance Continuous',

           'Immune.Bcell Pearson',
           'Immune.Bcell Pearson p-Value',
           'Immune.Bcell Spearman',
           'Immune.Bcell Spearman p-Value',
           'Immune.Bcell Entropy/Importance Continuous',

           'Immune.Mono Pearson',
           'Immune.Mono Pearson p-Value',
           'Immune.Mono Spearman',
           'Immune.Mono Spearman p-Value',
           'Immune.Mono Entropy/Importance Continuous',

           'Immune.Neutrophil Pearson',
           'Immune.Neutrophil Pearson p-Value',
           'Immune.Neutrophil Spearman',
           'Immune.Neutrophil Spearman p-Value',
           'Immune.Neutrophil Entropy/Importance Continuous',


           # Start of categorical variables
           'GrimAgePC T-Test',
           'GrimAgePC T-Test p-Value',
           'GrimAgePC ANOVA',
           'GrimAgePC ANOVA p-Value',
           'GrimAgePC Kruskal-Wallis',
           'GrimAgePC Kruskal-Wallis p-Value',
           'GrimAgePC Entropy/Importance Categorical',

           'PhenoAgePC T-Test',
           'PhenoAgePC T-Test p-Value',
           'PhenoAgePC ANOVA',
           'PhenoAgePC ANOVA p-Value',
           'PhenoAgePC Kruskal-Wallis',
           'PhenoAgePC Kruskal-Wallis p-Value',
           'PhenoAgePC Entropy/Importance Categorical',

           'HorvathPC T-Test',
           'HorvathPC T-Test p-Value',
           'HorvathPC ANOVA',
           'HorvathPC ANOVA p-Value',
           'HorvathPC Kruskal-Wallis',
           'HorvathPC Kruskal-Wallis p-Value',
           'HorvathPC Entropy/Importance Categorical',

           'HannumPC T-Test',
           'HannumPC T-Test p-Value',
           'HannumPC ANOVA',
           'HannumPC ANOVA p-Value',
           'HannumPC Kruskal-Wallis',
           'HannumPC Kruskal-Wallis p-Value',
           'HannumPC Entropy/Importance Categorical',

           'TelomerePC T-Test',
           'TelomerePC T-Test p-Value',
           'TelomerePC ANOVA',
           'TelomerePC ANOVA p-Value',
           'TelomerePC Kruskal-Wallis',
           'TelomerePC Kruskal-Wallis p-Value',
           'TelomerePC Entropy/Importance Categorical',

           'DunedinPACE T-Test',
           'DunedinPACE T-Test p-Value',
           'DunedinPACE ANOVA',
           'DunedinPACE ANOVA p-Value',
           'DunedinPACE Kruskal-Wallis',
           'DunedinPACE Kruskal-Wallis p-Value',
           'DunedinPACE Entropy/Importance Categorical',

           # 'AltumAge T-Test',
           # 'AltumAge T-Test p-Value',
           # 'AltumAge ANOVA',
           # 'AltumAge ANOVA p-value',
           # 'AltumAge Kruskal-Wallis',
           # 'AltumAge Kruskal-Wallis p-Value',
           # 'AltumAge Entropy/Importance Categorical'

           'Full_CumulativeStemCellDivisions T-Test',
           'Full_CumulativeStemCellDivisions T-Test p-Value',
           'Full_CumulativeStemCellDivisions ANOVA',
           'Full_CumulativeStemCellDivisions ANOVA p-Value',
           'Full_CumulativeStemCellDivisions Kruskal-Wallis',
           'Full_CumulativeStemCellDivisions Kruskal-Wallis p-Value',
           'Full_CumulativeStemCellDivisions Entropy/Importance Categorical',

           'Avg_LifeTime_IntrinsicStemCellDivisionRate T-Test',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate T-Test p-Value',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate ANOVA',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate ANOVA p-Value',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Kruskal-Wallis',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Kruskal-Wallis p-Value',
           'Avg_LifeTime_IntrinsicStemCellDivisionRate Entropy/Importance Categorical',

           'Median_Lifetime_IntrinsicStemCellDivisionRate T-Test',
           'Median_Lifetime_IntrinsicStemCellDivisionRate T-Test p-Value',
           'Median_Lifetime_IntrinsicStemCellDivisionRate ANOVA',
           'Median_Lifetime_IntrinsicStemCellDivisionRate ANOVA p-Value',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Kruskal-Wallis',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Kruskal-Wallis p-Value',
           'Median_Lifetime_IntrinsicStemCellDivisionRate Entropy/Importance Categorical',

           'Immune.CD8T T-Test',
           'Immune.CD8T T-Test p-Value',
           'Immune.CD8T ANOVA',
           'Immune.CD8T ANOVA p-Value',
           'Immune.CD8T Kruskal-Wallis',
           'Immune.CD8T Kruskal-Wallis p-Value',
           'Immune.CD8T Entropy/Importance Categorical',

           'Immune.CD4T T-Test',
           'Immune.CD4T T-Test p-Value',
           'Immune.CD4T ANOVA',
           'Immune.CD4T ANOVA p-Value',
           'Immune.CD4T Kruskal-Wallis',
           'Immune.CD4T Kruskal-Wallis p-Value',
           'Immune.CD4T Entropy/Importance Categorical',

           'Immune.CD4T.CD8T T-Test',
           'Immune.CD4T.CD8T T-Test p-Value',
           'Immune.CD4T.CD8T ANOVA',
           'Immune.CD4T.CD8T ANOVA p-Value',
           'Immune.CD4T.CD8T Kruskal-Wallis',
           'Immune.CD4T.CD8T Kruskal-Wallis p-Value',
           'Immune.CD4T.CD8T Entropy/Importance Categorical',

           'Immune.NK T-Test',
           'Immune.NK T-Test p-Value',
           'Immune.NK ANOVA',
           'Immune.NK ANOVA p-Value',
           'Immune.NK Kruskal-Wallis',
           'Immune.NK Kruskal-Wallis p-Value',
           'Immune.NK Entropy/Importance Categorical',

           'Immune.Bcell T-Test',
           'Immune.Bcell T-Test p-Value',
           'Immune.Bcell ANOVA',
           'Immune.Bcell ANOVA p-Value',
           'Immune.Bcell Kruskal-Wallis',
           'Immune.Bcell Kruskal-Wallis p-Value',
           'Immune.Bcell Entropy/Importance Categorical',

           'Immune.Mono T-Test',
           'Immune.Mono T-Test p-Value',
           'Immune.Mono ANOVA',
           'Immune.Mono ANOVA p-Value',
           'Immune.Mono Kruskal-Wallis',
           'Immune.Mono Kruskal-Wallis p-Value',
           'Immune.Mono Entropy/Importance Categorical',

           'Immune.Neutrophil T-Test',
           'Immune.Neutrophil T-Test p-Value',
           'Immune.Neutrophil ANOVA',
           'Immune.Neutrophil ANOVA p-Value',
           'Immune.Neutrophil Kruskal-Wallis',
           'Immune.Neutrophil Kruskal-Wallis p-Value',
           'Immune.Neutrophil Entropy/Importance Categorical'
           ]

combos = ['Ethnicity',
          'Cardiovascular',
          'Respiratory Disease',
          'Endocrine Disease',
          'Gastrointestinal',
          'Genito-Urinary',
          'Musculoskeletal',
          'Neuropsychological',
          'Reproductive',
          'Immune',
          'Do you take any of the Following nutritional Supplements?',
          'Do you take any of the following supplements or medications?',
          'Recreational Drug Use?',
          'What does your diet mostly consist of?'
          ]

# new_binaries = []
# for cov in combos:
#     for resp in covariates[cov]:
#         try:
#             if str(resp) != 'nan':
#                 all_resp = list(resp.split(';'))
#                 if "''" in all_resp:
#                     all_resp = all_resp.remove("''")
#                 elif '"' in all_resp:
#                     all_resp = all_resp.remove('"')
#                 if len(all_resp) > 1:
#                     for x in range(len(all_resp)):
#                         all_resp[x] = all_resp[x].lstrip()
#                         if '"' in all_resp[x]:
#                             all_resp[x] = all_resp[x].replace('"', '')
#                     # print(all_resp)
#                     for each in all_resp:
#                         if each not in new_binaries:
#                             new_binaries.append(resp)
#         except Exception as e:
#             print(e, resp)

# cols = covariates.columns
# pats = covariates.index
# total = len(new_binaries) * len(pats) * len(cols)
# x = 0
# for nb in new_binaries:
#     print(x / total)
#     for patient in pats:
#         for cov in cols:
#             if nb in str(covariates.loc[patient, cov]):
#                 covariates.loc[patient, nb] = 1
#             else:
#                 covariates.loc[patient, nb] = 0
#             x += 1
# covariates.to_csv('MetaDataBinaryAdded.csv')
# exit()

all_covariates = continuous+categorical
for c in range(len(all_covariates)):
    all_covariates[c] = all_covariates[c].lower()

drugs = pd.read_csv('Binary Stuff/RecDrugUse_CovariateData.csv')
meds = pd.read_csv('Binary Stuff/Medicine_CovariateData.csv')
supps = pd.read_csv('Binary Stuff/NutritionalSupplements_CovariateData.csv')
anti_aging = pd.read_csv('Binary Stuff/AntiAging - CovariateData.csv')
# meds.drop(columns=['NO NAD', 'NO NR', 'NO NMN'], inplace=True)
meds.drop(index=[0], inplace=True)
drugs.drop(index=[0, 1], inplace=True)
supps.drop(index=[0], inplace=True)

ind_cols = ['Total Respondents',
            'Avg Chronological Age',
            'Avg GrimAgePC',
            'Avg PhenoAgePC',
            'Avg HorvathPC',
            'Avg HannumPC',
            'Avg TelomerePC',
            'Avg DunedinPACE',
            'Avg Immune.CD8T',
            'Avg Immune.CD4T',
            'Avg Immune.CD4T.CD8T',
            'Avg Immune.NK',
            'Avg Immune.Bcell',
            'Avg Immune.Mono',
            'Avg Immune.Neutrophil',

            'GrimAgePC t-test',
            'GrimAgePC t-test p-Value',
            'GrimAgePC Kruskal-Wallis',
            'GrimAgePC Kruskal-Wallis p-Value',
            'PhenoAgePC t-test',
            'PhenoAgePC t-test p-Value',
            'PhenoAgePC Kruskal-Wallis',
            'PhenoAgePC Kruskal-Wallis p-Value',
            'HorvathPC t-test',
            'HorvathPC t-test p-Value',
            'HorvathPC Kruskal-Wallis',
            'HorvathPC Kruskal-Wallis p-Value',
            'HannumPC t-test',
            'HannumPC t-test p-Value',
            'HannumPC Kruskal-Wallis',
            'HannumPC Kruskal-Wallis p-Value',
            'DunedinPACE t-test',
            'DunedinPACE t-test p-Value',
            'DunedinPACE Kruskal-Wallis',
            'DunedinPACE Kruskal-Wallis p-Value',
            'TelomerePC t-test',
            'TelomerePC t-test p-Value',
            'TelomerePC Kruskal-Wallis',
            'TelomerePC Kruskal-Wallis p-Value',
            'Immune.CD8T t-test',
            'Immune.CD8T t-test p-Value',
            'Immune.CD8T Kruskal-Wallis',
            'Immune.CD8T Kruskal-Wallis p-Value',
            'Immune.CD4T t-test',
            'Immune.CD4T t-test p-Value',
            'Immune.CD4T Kruskal-Wallis',
            'Immune.CD4T Kruskal-Wallis p-Value',
            'Immune.CD4T.CD8T t-test',
            'Immune.CD4T.CD8T t-test p-Value',
            'Immune.CD4T.CD8T Kruskal-Wallis',
            'Immune.CD4T.CD8T Kruskal-Wallis p-Value',
            'Immune.NK t-test',
            'Immune.NK t-test p-Value',
            'Immune.NK Kruskal-Wallis',
            'Immune.NK Kruskal-Wallis p-Value',
            'Immune.Bcell t-test',
            'Immune.Bcell t-test p-Value',
            'Immune.Bcell Kruskal-Wallis',
            'Immune.Bcell Kruskal-Wallis p-Value',
            'Immune.Mono t-test',
            'Immune.Mono t-test p-Value',
            'Immune.Mono Kruskal-Wallis',
            'Immune.Mono Kruskal-Wallis p-Value',
            'Immune.Neutrophil t-test',
            'Immune.Neutrophil t-test p-Value',
            'Immune.Neutrophil Kruskal-Wallis',
            'Immune.Neutrophil Kruskal-Wallis p-Value']

associations = pd.DataFrame(columns=og_cols)
cat_associations = pd.DataFrame(columns=ind_cols)
binary_exclude_associations = pd.DataFrame(columns=ind_cols)

print('Length of ass. columns: ', len(associations.columns))
print('Length of cat-ass. columns: ', len(cat_associations.columns))

if new_uniques:
    uniques = pd.DataFrame(index=covariates.columns[1:], columns=['Uniques'])
    for feature in covariates.columns[1:]:
        unique_vals = np.unique(covariates[feature].astype(str))
        uniques.loc[feature, 'Uniques'] = unique_vals
        print('\nUniques for ', feature, unique_vals)
    uniques.to_csv('UniqueVals.csv')
else:
    uniques = pd.read_csv('UniqueVals.csv').set_index('Unnamed: 0')
print('Number of differential values: ', len(differentials.index))

if run_continuous:
    for feature in continuous[:]:
        print(feature)
        vals = pd.Series(covariates.loc[:, feature]).dropna()

        for v in range(len(vals)):
            value = list(vals)[v]
            if ',' in str(value):
                vals.replace(value, str(value).replace(',', '.'), inplace=True)
        try:
            vals.drop(index=vals[vals == 'remove this value'].index, inplace=True)
        except:
            pass

        vals.replace(' ', np.nan, inplace=True)
        vals.replace('', np.nan, inplace=True)
        vals.replace('2.417.71', np.nan, inplace=True)
        vals = vals.astype(float)

        if feature == 'Age for Loss of Hair':
            vals.replace(0, np.nan, inplace=True)

        patients = (set(vals.index) & set(differentials.index))
        diffs = differentials.loc[patients]
        vals = vals.loc[patients]

        remove_indices = vals.loc[vals.isnull()].index
        vals.drop(index=remove_indices, inplace=True)
        diffs.drop(index=remove_indices, inplace=True)
        patients = (set(vals.index) & set(differentials.index))
        print('Length of values: ', len(vals))

        GrimCorr, GrimP = stats.pearsonr(vals, diffs.loc[patients, 'GrimAgePC'])
        PhenoCorr, PhenoP = stats.pearsonr(vals, diffs.loc[patients, 'PhenoAgePC'])
        HorPC_Corr, HorPC_P = stats.pearsonr(vals, diffs.loc[patients, 'HorvathPC'])
        HanPC_Corr, HanPC_P = stats.pearsonr(vals, diffs.loc[patients, 'HannumPC'])
        teloPC_Corr, teloPC_P = stats.pearsonr(vals, diffs.loc[patients, 'TelomerePC'])
        duneCorr, duneP = stats.pearsonr(vals, diffs.loc[patients, 'DunedinPACE'])
        # altumCorr, altumP = stats.pearsonr(vals, diffs.loc[patients, 'AltumAge'])
        fullCellCorr, fullCellP = stats.pearsonr(vals, diffs.loc[patients, 'Full_CumulativeStemCellDivisions'])
        avgCellCorr, avgCellP = stats.pearsonr(vals, diffs.loc[patients, 'Avg_LifeTime_IntrinsicStemCellDivisionRate'])
        medianCellCorr, medianCellP = stats.pearsonr(vals, diffs.loc[patients, 'Median_Lifetime_IntrinsicStemCellDivisionRate'])
        cd8Corr, cd8P = stats.pearsonr(vals, diffs.loc[patients, 'Immune.CD8T'])
        cd4Corr, cd4P = stats.pearsonr(vals, diffs.loc[patients, 'Immune.CD4T'])
        cdRatioCorr, cdRatioP = stats.pearsonr(vals, diffs.loc[patients, 'Immune.CD4T.CD8T'])
        nkCorr, nkP = stats.pearsonr(vals, diffs.loc[patients, 'Immune.NK'])
        bCellCorr, bCellP = stats.pearsonr(vals, diffs.loc[patients, 'Immune.Bcell'])
        monoCorr, monoP = stats.pearsonr(vals, diffs.loc[patients, 'Immune.Mono'])
        neutroCorr, neutroP = stats.pearsonr(vals, diffs.loc[patients, 'Immune.Neutrophil'])

        GrimSpear, GrimS_P = stats.spearmanr(vals, diffs.loc[patients, 'GrimAgePC'])
        PhenoSpear, PhenoS_P = stats.spearmanr(vals, diffs.loc[patients, 'PhenoAgePC'])
        HorPC_Spear, HorPC_S_P = stats.spearmanr(vals, diffs.loc[patients, 'HorvathPC'])
        HanPC_Spear, HanPC_S_P = stats.spearmanr(vals, diffs.loc[patients, 'HannumPC'])
        teloPC_Spear, teloPC_S_P = stats.spearmanr(vals, diffs.loc[patients, 'TelomerePC'])
        duneSpear, duneS_P = stats.spearmanr(vals, diffs.loc[patients, 'DunedinPACE'])
        # altumSpear, altumS_P = stats.spearmanr(vals, diffs.loc[patients, 'AltumAge'])
        fullCellSpear, fullCellS_P = stats.spearmanr(vals, diffs.loc[patients, 'Full_CumulativeStemCellDivisions'])
        avgCellSpear, avgCellS_P = stats.spearmanr(vals, diffs.loc[patients, 'Avg_LifeTime_IntrinsicStemCellDivisionRate'])
        medianCellSpear, medianCellS_P = stats.spearmanr(vals, diffs.loc[patients, 'Median_Lifetime_IntrinsicStemCellDivisionRate'])
        cd8Spear, cd8S_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.CD8T'])
        cd4Spear, cd4S_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.CD4T'])
        cdRatioSpear, cdRatioS_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.CD4T.CD8T'])
        nkSpear, nkS_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.NK'])
        bCellSpear, bCellS_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.Bcell'])
        monoSpear, monoS_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.Mono'])
        neutroSpear, neutroS_P = stats.spearmanr(vals, diffs.loc[patients, 'Immune.Neutrophil'])

        GrimEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'GrimAgePC'], discrete_features=False)[0]
        PhenoEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'PhenoAgePC'], discrete_features=False)[0]
        HorPC_Entropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'HorvathPC'], discrete_features=False)[0]
        HanPC_Entropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'HannumPC'], discrete_features=False)[0]
        teloPC_Entropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'TelomerePC'], discrete_features=False)[0]
        duneEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'DunedinPACE'], discrete_features=False)[0]
        # altumEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'AltumAge'], discrete_features=False)[0]
        fullCellEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Full_CumulativeStemCellDivisions'], discrete_features=False)[0]
        avgCellEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Avg_LifeTime_IntrinsicStemCellDivisionRate'], discrete_features=False)[0]
        medianCellEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Median_Lifetime_IntrinsicStemCellDivisionRate'], discrete_features=False)[0]
        cd8Entropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.CD8T'], discrete_features=False)[0]
        cd4Entropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.CD4T'], discrete_features=False)[0]
        cdRatioEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.CD4T.CD8T'], discrete_features=False)[0]
        nkEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.NK'], discrete_features=False)[0]
        bCellEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.Bcell'], discrete_features=False)[0]
        monoEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.Mono'], discrete_features=False)[0]
        neutroEntropy = mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.Neutrophil'], discrete_features=False)[0]

        print('Sample full entropy output: ', mutual_info_regression(np.array(vals).reshape(-1, 1), diffs.loc[patients, 'Immune.Neutrophil'], discrete_features=False))

        associations.loc[feature, 'GrimAgePC Pearson'] = GrimCorr
        associations.loc[feature, 'GrimAgePC Pearson p-Value'] = GrimP
        associations.loc[feature, 'GrimAgePC Spearman'] = GrimSpear
        associations.loc[feature, 'GrimAgePC Spearman p-Value'] = GrimS_P
        associations.loc[feature, 'GrimAgePC Entropy/Importance Continuous'] = GrimEntropy

        associations.loc[feature, 'PhenoAgePC Pearson'] = PhenoCorr
        associations.loc[feature, 'PhenoAgePC Pearson p-Value'] = PhenoP
        associations.loc[feature, 'PhenoAgePC Spearman'] = PhenoSpear
        associations.loc[feature, 'PhenoAgePC Spearman p-Value'] = PhenoS_P
        associations.loc[feature, 'PhenoAgePC Entropy/Importance Continuous'] = PhenoEntropy

        associations.loc[feature, 'HorvathPC Pearson'] = HorPC_Corr
        associations.loc[feature, 'HorvathPC Pearson p-Value'] = HorPC_P
        associations.loc[feature, 'HorvathPC Spearman'] = HorPC_Spear
        associations.loc[feature, 'HorvathPC Spearman p-Value'] = HorPC_S_P
        associations.loc[feature, 'HorvathPC Entropy/Importance Continuous'] = HorPC_Entropy

        associations.loc[feature, 'HannumPC Pearson'] = HanPC_Corr
        associations.loc[feature, 'HannumPC Pearson p-Value'] = HanPC_P
        associations.loc[feature, 'HannumPC Spearman'] = HanPC_Spear
        associations.loc[feature, 'HannumPC Spearman p-Value'] = HanPC_S_P
        associations.loc[feature, 'HannumPC Entropy/Importance Continuous'] = HanPC_Entropy

        associations.loc[feature, 'TelomerePC Pearson'] = teloPC_Corr
        associations.loc[feature, 'TelomerePC Pearson p-Value'] = teloPC_P
        associations.loc[feature, 'TelomerePC Spearman'] = teloPC_Spear
        associations.loc[feature, 'TelomerePC Spearman p-Value'] = teloPC_S_P
        associations.loc[feature, 'TelomerePC Entropy/Importance Continuous'] = teloPC_Entropy

        associations.loc[feature, 'DunedinPACE Pearson'] = duneCorr
        associations.loc[feature, 'DunedinPACE Pearson p-Value'] = duneP
        associations.loc[feature, 'DunedinPACE Spearman'] = duneSpear
        associations.loc[feature, 'DunedinPACE Spearman p-Value'] = duneS_P
        associations.loc[feature, 'DunedinPACE Entropy/Importance Continuous'] = duneEntropy

        # associations.loc[feature, 'AltumAge Pearson'] = altumCorr
        # associations.loc[feature, 'AltumAge Pearson p-Value'] = altumP
        # associations.loc[feature, 'AltumAge Spearman'] = altumSpear
        # associations.loc[feature, 'AltumAge Spearman p-Value'] = altumS_P
        # associations.loc[feature, 'AltumAge Entropy/Importance Continuous'] = altumEntropy

        associations.loc[feature, 'Full_CumulativeStemCellDivisions Pearson'] = fullCellCorr
        associations.loc[feature, 'Full_CumulativeStemCellDivisions Pearson p-Value'] = fullCellP
        associations.loc[feature, 'Full_CumulativeStemCellDivisions Spearman'] = fullCellSpear
        associations.loc[feature, 'Full_CumulativeStemCellDivisions Spearman p-Value'] = fullCellS_P
        associations.loc[feature, 'Full_CumulativeStemCellDivisions Entropy/Importance Continuous'] = fullCellEntropy

        associations.loc[feature, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Pearson'] = avgCellCorr
        associations.loc[feature, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Pearson p-Value'] = avgCellP
        associations.loc[feature, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Spearman'] = avgCellSpear
        associations.loc[feature, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Spearman p-Value'] = avgCellS_P
        associations.loc[feature, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Entropy/Importance Continuous'] = avgCellEntropy

        associations.loc[feature, 'Median_Lifetime_IntrinsicStemCellDivisionRate Pearson'] = medianCellCorr
        associations.loc[feature, 'Median_Lifetime_IntrinsicStemCellDivisionRate Pearson p-Value'] = medianCellP
        associations.loc[feature, 'Median_Lifetime_IntrinsicStemCellDivisionRate Spearman'] = medianCellSpear
        associations.loc[feature, 'Median_Lifetime_IntrinsicStemCellDivisionRate Spearman p-Value'] = medianCellS_P
        associations.loc[feature, 'Median_Lifetime_IntrinsicStemCellDivisionRate Entropy/Importance Continuous'] = medianCellEntropy

        associations.loc[feature, 'Immune.CD8T Pearson'] = cd8Corr
        associations.loc[feature, 'Immune.CD8T Pearson p-Value'] = cd8P
        associations.loc[feature, 'Immune.CD8T Spearman'] = cd8Spear
        associations.loc[feature, 'Immune.CD8T Spearman p-Value'] = cd8S_P
        associations.loc[feature, 'Immune.CD8T Entropy/Importance Continuous'] = cd8Entropy

        associations.loc[feature, 'Immune.CD4T Pearson'] = cd4Corr
        associations.loc[feature, 'Immune.CD4T Pearson p-Value'] = cd4P
        associations.loc[feature, 'Immune.CD4T Spearman'] = cd4Spear
        associations.loc[feature, 'Immune.CD4T Spearman p-Value'] = cd4S_P
        associations.loc[feature, 'Immune.CD4T Entropy/Importance Continuous'] = cd4Entropy

        associations.loc[feature, 'Immune.CD4T.CD8T Pearson'] = cdRatioCorr
        associations.loc[feature, 'Immune.CD4T.CD8T Pearson p-Value'] = cdRatioP
        associations.loc[feature, 'Immune.CD4T.CD8T Spearman'] = cdRatioSpear
        associations.loc[feature, 'Immune.CD4T.CD8T Spearman p-Value'] = cdRatioS_P
        associations.loc[feature, 'Immune.CD4T.CD8T Entropy/Importance Continuous'] = cdRatioEntropy

        associations.loc[feature, 'Immune.NK Pearson'] = nkCorr
        associations.loc[feature, 'Immune.NK Pearson p-Value'] = nkP
        associations.loc[feature, 'Immune.NK Spearman'] = nkSpear
        associations.loc[feature, 'Immune.NK Spearman p-Value'] = nkS_P
        associations.loc[feature, 'Immune.NK Entropy/Importance Continuous'] = nkEntropy

        associations.loc[feature, 'Immune.Bcell Pearson'] = bCellCorr
        associations.loc[feature, 'Immune.Bcell Pearson p-Value'] = bCellP
        associations.loc[feature, 'Immune.Bcell Spearman'] = bCellSpear
        associations.loc[feature, 'Immune.Bcell Spearman p-Value'] = bCellS_P
        associations.loc[feature, 'Immune.Bcell Entropy/Importance Continuous'] = bCellEntropy

        associations.loc[feature, 'Immune.Mono Pearson'] = monoCorr
        associations.loc[feature, 'Immune.Mono Pearson p-Value'] = monoP
        associations.loc[feature, 'Immune.Mono Spearman'] = monoSpear
        associations.loc[feature, 'Immune.Mono Spearman p-Value'] = monoS_P
        associations.loc[feature, 'Immune.Mono Entropy/Importance Continuous'] = monoEntropy

        associations.loc[feature, 'Immune.Neutrophil Pearson'] = neutroCorr
        associations.loc[feature, 'Immune.Neutrophil Pearson p-Value'] = neutroP
        associations.loc[feature, 'Immune.Neutrophil Spearman'] = neutroSpear
        associations.loc[feature, 'Immune.Neutrophil Spearman p-Value'] = neutroS_P
        associations.loc[feature, 'Immune.Neutrophil Entropy/Importance Continuous'] = neutroEntropy

        for clock in age_differentials.columns[1:]:
            vals.sort_values(ascending=True, inplace=True)
            ordered_patients = vals.index
            residuals = diffs.loc[ordered_patients, clock]
            plt.scatter(vals, residuals)
            plt.xlabel(feature)

            a, b = np.polyfit(vals, residuals, 1)
            plt.plot(vals, a*vals+b, 'r-')

            if clock == 'GrimAgePC':
                var_name = 'Grim'
            elif clock == 'PhenoAgePC':
                var_name = 'Pheno'
            elif clock == 'HorvathPC':
                var_name = 'HorPC_'
            elif clock == 'HannumPC':
                var_name = 'HanPC_'
            elif clock == 'TelomerePC':
                var_name = 'teloPC_'
            elif clock == 'DunedinPACE':
                var_name = 'dune'
            elif clock == 'AltumAge':
                var_name = 'altum'
            elif clock == 'Full_CumulativeStemCellDivisions':
                var_name = 'fullCell'
            elif clock == 'Avg_LifeTime_IntrinsicStemCellDivisionRate':
                var_name = 'avgCell'
            elif clock == 'Median_Lifetime_IntrinsicStemCellDivisionRate':
                var_name = 'medianCell'
            elif clock == 'Immune.CD8T':
                var_name = 'cd8'
            elif clock == 'Immune.CD4T':
                var_name = 'cd4'
            elif clock == 'Immune.CD4T.CD8T':
                var_name = 'cdRatio'
            elif clock == 'Immune.NK':
                var_name = 'nk'
            elif clock == 'Immune.Bcell':
                var_name = 'bCell'
            elif clock == 'Immune.Mono':
                var_name = 'mono'
            elif clock == 'Immune.Neutrophil':
                var_name = 'neutro'

            eval_string = var_name + 'S_P'
            eval_stringPears = var_name + 'P'
            trend = str(round(a, 5))
            spear_p = str(round(eval(eval_string), 4))
            pears_p = str(round(eval(eval_stringPears), 4))

            if clock in immunes:
                plt.title('Cell values for ' + clock + ', ' + feature + '\nSpearman p-Val: ' + spear_p
                          + '  || Pearson p-Val: ' + pears_p + ' || Trend: ' + trend)
            else:
                plt.title('Residuals for ' + clock + ', ' + feature + '\nSpearman p-Val: ' + spear_p
                          + ' || Pearson p-Val: ' + pears_p + ' || Trend: ' + trend)
            if save_figs:
                if float(spear_p) < .05 or float(pears_p) < .05:
                    plt.savefig('significant association figures/' + feature + '-' + clock + 'spear_p-' + spear_p + 'pears_p-' + pears_p + '.png')
                else:
                    plt.savefig('non-significant association figures/' + feature + '-' + clock + 'spear_p-' + spear_p + 'pears_p-' + pears_p + '.png')
            if show_continuous:
                plt.show()
            plt.close()

# Cleaning bad values


def anova(groups_, length, clock_):
    groupVals = groups_ + "['" + clock_ + "']"
    exec_string = 'scipy.stats.f_oneway('
    for i in range(length):
        if i != length - 1:
            exec_string += groupVals + '[' + str(i) + '],'
        else:
            exec_string += groupVals + '[' + str(i) + '])'
    return eval(exec_string)


def kruskal(groups_, length, clock_):
    groupVals = groups_ + "['" + clock_ + "']"
    exec_string = 'scipy.stats.kruskal('
    for i in range(length):
        if i != length - 1:
            exec_string += groupVals + '[' + str(i) + '],'
        else:
            exec_string += groupVals + '[' + str(i) + '])'
    return eval(exec_string)


def demo_split(f, feat_name, vs, clock_name):
    execute = True
    if feat_name.lower() == 'biological sex':
        execute = False

    if feat_name.lower() == 'menopause':
        box_titles = ['Women w Menopause, HRT vs. no HRT']
        all_demos = [['hrt', 'no hrt']]
        is_feats = {'anti-aging': [covariates[['hormone replacement therapy' in str(x).lower() for x in covariates['Actively engage in anti-aging interventions?']]],
                                   covariates[['hormone replacement therapy' not in str(x).lower() for x in covariates['Actively engage in anti-aging interventions?']]]]}
    elif feat_name.lower() == '':
        box_titles = ['']
        all_demos = ['', '']
        is_feats = {'': covariates[['' in str(x) for x in covariates['']]].index}
    elif f.lower() == 'supplements':
        box_titles = ['Diabetes vs None', 'Male vs. Female']
        all_demos = [['diabetes', 'no diabetes'], ['male', 'female']]
        is_feats = {'endocrine disease': [covariates[['diabetes' in str(x).lower() for x in covariates['Endocrine Disease']]],
                                          covariates[['diabetes' not in str(x).lower() for x in covariates['Endocrine Disease']]]],
                    'biological sex': [list(covariates[covariates['Biological Sex'] == 'male'].index),
                                       list(covariates[covariates['Biological Sex'] == 'female'].index)]}
    else:
        box_titles = ['Male vs Female']
        all_demos = [['male', 'female']]
        is_feats = {'biological sex': [list(covariates[covariates['Biological Sex'] == 'male'].index),
                                       list(covariates[covariates['Biological Sex'] == 'female'].index)]}

    if execute:
        for x in range(len(all_demos)):
            demos = all_demos[x]
            box_title = feat_name + box_titles[x]

            for k in range(len(is_feats.keys())):
                key = list(is_feats.keys())[k]
                for d in range(len(list(is_feats[key]))):
                    is_feat = is_feat[key[d]]
                    demo = demos[d]
                    is_feat = (set(is_feat) & set(vs.index))

                    ######

                    demo_residuals = {}

                    split_diffs = diffs.loc[is_feat]

                    for u in uniques:
                        unique_pats = vals[vals == u].index
                        shared = list(set(unique_pats) & set(split_diffs.index))
                        demo_residuals[u] = list(split_diffs.loc[shared, clock])

                    figa = plt.figure()
                    # fig.subplots_adjust(bottom=150, top=200)
                    box = figa.add_subplot()
                    box.boxplot(demo_residuals.values())

                    if featurecopy in ordinals:
                        avgs = [np.median(x) for x in demo_residuals.values()]

                        indices = list(range(len(demo_residuals.keys())))
                        indices = [x + 1 for x in indices]
                        a, b = np.polyfit(indices, avgs, 1)
                        box.plot(valcopy, a * valcopy + b, 'r-')
                        trend = str(round(a, 5))

                    box.set_xticklabels(demo_residuals.keys())

                    if clock_name == 'GrimAgePC':
                        var_name = 'Grim'
                    elif clock_name == 'PhenoAgePC':
                        var_name = 'Pheno'
                    elif clock_name == 'HorvathPC':
                        var_name = 'HorPC_'
                    elif clock_name == 'HannumPC':
                        var_name = 'HanPC_'
                    elif clock_name == 'TelomerePC':
                        var_name = 'teloPC_'
                    elif clock_name == 'DunedinPACE':
                        var_name = 'dune'
                    elif clock_name == 'AltumAge':
                        var_name = 'altum'
                    elif clock_name == 'Full_CumulativeStemCellDivisions':
                        var_name = 'fullCell'
                    elif clock_name == 'Avg_LifeTime_IntrinsicStemCellDivisionRate':
                        var_name = 'avgCell'
                    elif clock_name == 'Median_Lifetime_IntrinsicStemCellDivisionRate':
                        var_name = 'medianCell'
                    elif clock_name == 'Immune.CD8T':
                        var_name = 'cd8'
                    elif clock_name == 'Immune.CD4T':
                        var_name = 'cd4'
                    elif clock_name == 'Immune.CD4T.CD8T':
                        var_name = 'cdRatio'
                    elif clock_name == 'Immune.NK':
                        var_name = 'nk'
                    elif clock_name == 'Immune.Bcell':
                        var_name = 'bCell'
                    elif clock_name == 'Immune.Mono':
                        var_name = 'mono'
                    elif clock_name == 'Immune.Neutrophil':
                        var_name = 'neutro'

                    eval_string = var_name + 'KK_P'
                    eval_stringANOVA = var_name + 'ANOVA_P'
                    kk_p = str(round(eval(eval_string), 4))
                    ANOVA_P = str(round(eval(eval_stringANOVA), 4))

                    # Creating the text addendum at the bottom of the figure
                    text_string = "plt.figtext(0.5, 0.001, 'Set includes ' + demo + ' only \\n' + 'Sample Size: ' + str("
                    for x in range(len(list(demo_residuals.values()))):
                        size = str(len(list(demo_residuals.values())[x]))
                        if x != len(list(demo_residuals.values()))-1:
                            text_string += ' + ' + size
                        else:
                            text_string += ' + ' + size + '), horizontalalignment="center")'
                    exec(text_string)

                    if clock_name in immunes:
                        if featurecopy in ordinals:
                            box.set_title(
                                'Cell values for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                        else:
                            box.set_title('Cell values for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
                    else:
                        if clock_name == 'TelomerePC':
                            if featurecopy in ordinals:
                                box.set_title(
                                    'Lengths for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                            else:
                                box.set_title('Lengths for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
                        elif clock_name == 'DunedinPACE':
                            if featurecopy in ordinals:
                                box.set_title('Aging Rate for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p
                                              + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                            else:
                                box.set_title('Aging Rate for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
                        else:
                            if featurecopy in ordinals:
                                box.set_title('Residuals for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p
                                              + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                            else:
                                box.set_title('Residuals for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + kk_p)

                    if save_figs:
                        if float(kk_p) < .05 or float(ANOVA_P) < .05:
                            plt.savefig('significant association figures/' + feat + '-' + clock_name + '-' + demo + ' ONLY.png')
                        else:
                            plt.savefig('non-significant association figures/' + feat + '-' + clock_name + '-' + demo + ' ONLY.png')
                    if show_residuals:
                        plt.show()

                    plt.close()

                    ######

            #         # vs = vs.loc[is_feat]
            #
            #         feat_pos_diffs = vs.loc[is_feat, c]
            #         feat_neg_diffs = vs.drop(index=is_feat)[c]
            #         print(vs)
            #         # print('Feat pos: ', feat_pos_diffs, '\nfeat neg: ', feat_neg_diffs)
            #         # print('Is feat: ', is_feat, '\nOther stuff: ', c, vs)
            #
            #         kVal, k_P_Val = stats.kruskal(feat_pos_diffs, feat_neg_diffs)
            #         tVal, tt_p_val = stats.ttest_ind(feat_pos_diffs, feat_neg_diffs, equal_var=False)
            #
            #         figa = plt.figure()
            #         box = figa.add_subplot()
            #         # fig.subplots_adjust(bottom=150, top=200)
            #         box.boxplot([feat_pos_diffs, feat_neg_diffs])
            #
            #         # exec_string = 'box.set_xticklabels(['
            #         # for d in range(len(demos)):
            #         #     if d == len(demos)-1:
            #         #         exec_string += 'demos[' + str(d) + ']])'
            #         #     else:
            #         #         exec_string += 'demos[' + str(d) + '], '
            #         # exec(exec_string)
            #
            #         avgs = [np.median(x) for x in [feat_pos_diffs, feat_neg_diffs]]
            #         a, b = np.polyfit([1, 2], avgs, 1)
            #         box.plot(pd.Series([1, 2]), pd.Series([1, 2]) * a + b, 'r-')
            #
            #         box.set_title('Residuals for ' + clock_name + ', ' + box_title + '\n Kruskal p-Val: ' + str(
            #                       round(k_P_Val, 4)) + ' || T-test p-Val: ' + str(round(tt_p_val, 4)) + ' || Trend: ' + str(round(a, 5)))
            #
            #         if show_residuals:
            #             plt.show()
            #         if save_figs:
            #             box_title = box_title.replace('?', '').replace('/', '')
            #             if float(k_P_Val) < .05 or float(str(tt_p_val)) < .05:
            #                 plt.savefig('significant association figures/' + feat + '-' + clock_name + '-' + box_title + '.png')
            #             else:
            #                 plt.savefig('non-significant association figures/' + feat + '-' + clock_name + '-' + box_title + '.png')
            #         plt.close()
            #
            #     if list(feat_neg_diffs) == list(feat_pos_diffs):
            #         pass
            #     else:
            #         if un == demos[0]:  # standard should go here (first in demos)
            #             tt, ttp = scipy.stats.ttest_ind(feat_pos_diffs, feat_neg_diffs, equal_var=False)
            #             try:
            #                 kk, kkp = stats.kruskal(feat_pos_diffs, feat_neg_diffs)
            #             except:
            #                 kk, kkp = np.nan, np.nan
            #
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], 'Total Respondents'] = len(feat_pos_diffs)
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], 'Avg ' + clock_name] = np.mean(feat_pos_diffs)
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' t-test'] = tt
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' t-test p-Value'] = ttp
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' Kruskal-Wallis'] = kk
            #             cat_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' Kruskal-Wallis p-Value'] = kkp
            #
            #     if un == demos[0]:
            #         unique_residuals = diffs.loc[standard_patients, clock_name]
            #         binary_excluded_residuals = diffs.drop(index=unique_patients)[clock_name]
            #
            #         binary_excluded_residuals = feat_pos_diffs
            #         unique_residuals = feat_neg_diffs
            #     else:
            #         binary_excluded_residuals = feat_neg_diffs
            #         unique_residuals = feat_pos_diffs
            #
            #     demo_residuals[demos[0]] = feat_pos_diffs
            #     demo_residuals[demos[1]] = feat_neg_diffs
            #
            #     print(feat_pos_diffs, feat_neg_diffs)
            #
            #     be_tt, be_ttp = scipy.stats.ttest_ind(unique_residuals, binary_excluded_residuals, equal_var=False)
            #     try:
            #         be_kk, be_kkp = stats.kruskal(unique_residuals, binary_excluded_residuals)
            #     except:
            #         be_kk, be_kkp = np.nan, np.nan
            #
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], 'Total Respondents'] = len(unique_residuals)
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], 'Avg ' + clock_name] = np.mean(unique_residuals)
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' t-test'] = be_tt
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' t-test p-Value'] = be_ttp
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' Kruskal-Wallis'] = be_kk
            #     binary_exclude_associations.loc[f + ': ' + un + ' | ' + demos[0] + 'vs. ' + demos[1], clock_name + ' Kruskal-Wallis p-Value'] = be_kkp
            #
            # figa = plt.figure()
            # # fig.subplots_adjust(bottom=150, top=200)
            # box = figa.add_subplot()
            # print(demo_residuals)
            # box.boxplot(demo_residuals.values())
            #
            # if featurecopy in ordinals:
            #     avgs = [np.median(x) for x in demo_residuals.values()]
            #
            #     indices = list(range(len(demo_residuals.keys())))
            #     indices = [x+1 for x in indices]
            #     a, b = np.polyfit(indices, avgs, 1)
            #     box.plot(valcopy, a * valcopy + b, 'r-')
            #
            # box.set_xticklabels(demo_residuals.keys())
            #
            # if featurecopy in ordinals:
            #     trend = str(round(a, 5))
            #
            # if clock_name in immunes:
            #     if featurecopy in ordinals:
            #         box.set_title('Cell values for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)) + ' || T-Test p-Val: ' + str(round(be_ttp, 4)) + ' || Trend: ' + trend)
            #     else:
            #         box.set_title('Cell values for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)))
            # else:
            #     if clock_name == 'TelomerePC':
            #         if featurecopy in ordinals:
            #             box.set_title('Lengths for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)) + ' || T-Test p-Val: ' + str(round(be_ttp, 4)) + ' || Trend: ' + trend)
            #         else:
            #             box.set_title('Lengths for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)))
            #     elif clock_name == 'DunedinPACE':
            #         if featurecopy in ordinals:
            #             box.set_title('Aging Rate for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4))
            #                           + ' || T-Test p-Val: ' + str(round(be_ttp, 4)) + ' || Trend: ' + trend)
            #         else:
            #             box.set_title('Aging Rate for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)))
            #     else:
            #         if featurecopy in ordinals:
            #             box.set_title('Residuals for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4))
            #                           + ' || T-Test p-Val: ' + str(round(be_ttp, 4)) + ' || Trend: ' + trend)
            #         else:
            #             box.set_title('Residuals for ' + clock_name + ', ' + feat + '\nKruskal p-Val: ' + str(round(be_kkp, 4)))
            # if save_figs:
            #     if float(be_kkp) < .05 or float(str(be_ttp)) < .05:
            #         plt.savefig('significant association figures/' + feat + '-' + clock_name + '.png')
            #     else:
            #         plt.savefig('non-significant association figures/' + feat + '-' + clock_name + '.png')
            # if show_residuals:
            #     plt.show()
            #
            # plt.close()


s_indices = (set(drugs['None']) & set(differentials.index))
print(drugs, meds)
for option in drugs.columns[1:]:
    for clock in age_differentials.columns[1:]:
        feat = 'Recreational Drugs'

        indices = (set(drugs[option].dropna()) & set(differentials.index))
        standard = differentials.loc[s_indices, clock]
        resp_residuals = differentials.loc[indices, clock]

        if len(resp_residuals.values) > 1:
            tt, tt_p = stats.ttest_ind(resp_residuals, standard, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, standard)

            cat_associations.loc['Recreational Drugs: ' + option, 'Total Respondents'] = len(resp_residuals)
            cat_associations.loc['Recreational Drugs: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            cat_associations.loc['Recreational Drugs: ' + option, clock + ' t-test'] = tt
            cat_associations.loc['Recreational Drugs: ' + option, clock + ' t-test p-Value'] = tt_p
            cat_associations.loc['Recreational Drugs: ' + option, clock + ' Kruskal-Wallis'] = kk
            cat_associations.loc['Recreational Drugs: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            excluded = differentials.drop(index=s_indices)[clock]
            num_samples = len(resp_residuals)

            tt, tt_p = stats.ttest_ind(resp_residuals, excluded, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, excluded)

            binary_exclude_associations.loc['Recreational Drugs: ' + option, 'Total Respondents'] = len(resp_residuals)
            binary_exclude_associations.loc['Recreational Drugs: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            binary_exclude_associations.loc['Recreational Drugs: ' + option, clock + ' t-test'] = tt
            binary_exclude_associations.loc['Recreational Drugs: ' + option, clock + ' t-test p-Value'] = tt_p
            binary_exclude_associations.loc['Recreational Drugs: ' + option, clock + ' Kruskal-Wallis'] = kk
            binary_exclude_associations.loc['Recreational Drugs: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            fig = plt.figure()
            # fig.subplots_adjust(bottom=150, top=200)
            box = fig.add_subplot()
            box.boxplot([resp_residuals, excluded])

            avgs = [np.median(x) for x in [resp_residuals, excluded]]
            avg1 = round(np.average(resp_residuals), 2)
            avg2 = round(np.average(excluded), 2)

            indices = [0, 1]
            a, b = np.polyfit(indices, avgs, 1)
            # box.plot(valcopy, a * valcopy + b, 'r-')
            # print(valcopy, a * valcopy + b)

            box.set_xticklabels([option, 'Remainder of cohort'])

            trend = str(round(a, 5))

            if len(option) > 20:
                new_u = option[:20]
            else:
                new_u = option

            plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(resp_residuals)) + ', ' +
                        str(len(excluded)), horizontalalignment='center')

            if clock in immunes:
                if option.lower() in ordinals:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                else:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
            else:
                if clock == 'TelomerePC':
                    if option.lower() in ordinals:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5)) + ', ' + str(round(tt_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                    else:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                elif clock == 'DunedinPACE':
                    if option.lower() in ordinals:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                else:
                    if option.lower() in ordinals:
                        box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title(
                            'BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p:' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
            if save_figs and num_samples > 15:
                if float(kk_p) < .05 or float(str(tt_p)) < .05:
                    plt.savefig('significant association figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                else:
                    plt.savefig('Non-Significant Association Figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                    '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')

# s_indices = (set(meds['No HRT']) & set(differentials.index))
# so_indices = (set(meds['None of These']) & set(differentials.index))

s_indices = (set(meds['Blank/none']) & set(differentials.index))

for option in meds.columns[:]:
    for clock in age_differentials.columns[1:]:
        feat = 'Medications'
        indices = (set(meds[option].dropna()) & set(differentials.index))
        resp_residuals = differentials.loc[indices, clock]

        if len(resp_residuals.values) > 1:
            standard = differentials.loc[s_indices, clock]

            tt, tt_p = stats.ttest_ind(resp_residuals, standard, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, standard)

            cat_associations.loc['Medications: ' + option, 'Total Respondents'] = len(resp_residuals)
            cat_associations.loc['Medications: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            cat_associations.loc['Medications: ' + option, clock + ' t-test'] = tt
            cat_associations.loc['Medications: ' + option, clock + ' t-test p-Value'] = tt_p
            cat_associations.loc['Medications: ' + option, clock + ' Kruskal-Wallis'] = kk
            cat_associations.loc['Medications: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            excluded = differentials.drop(index=s_indices)[clock]
            num_samples = len(resp_residuals)

            tt, tt_p = stats.ttest_ind(resp_residuals, excluded, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, excluded)

            binary_exclude_associations.loc['Medications: ' + option, 'Total Respondents'] = len(resp_residuals)
            binary_exclude_associations.loc['Medications: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            binary_exclude_associations.loc['Medications: ' + option, clock + ' t-test'] = tt
            binary_exclude_associations.loc['Medications: ' + option, clock + ' t-test p-Value'] = tt_p
            binary_exclude_associations.loc['Medications: ' + option, clock + ' Kruskal-Wallis'] = kk
            binary_exclude_associations.loc['Medications: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            fig = plt.figure()
            # fig.subplots_adjust(bottom=150, top=200)
            box = fig.add_subplot()
            box.boxplot([resp_residuals, excluded])

            avgs = [np.median(x) for x in [resp_residuals, excluded]]
            avg1 = round(np.average(resp_residuals), 2)
            avg2 = round(np.average(excluded), 2)

            indices = [0, 1]
            a, b = np.polyfit(indices, avgs, 1)
            # box.plot(valcopy, a * valcopy + b, 'r-')
            # print(valcopy, a * valcopy + b)

            box.set_xticklabels([option, 'Remainder of cohort'])

            trend = str(round(a, 5))

            if len(option) > 20:
                new_u = option[:20]
            else:
                new_u = option

            plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(resp_residuals)) + ', ' +
                        str(len(excluded)), horizontalalignment='center')

            if clock in immunes:
                if option.lower() in ordinals:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
                else:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
            else:
                if clock == 'TelomerePC':
                    if option.lower() in ordinals:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5)) + ', ' + str(round(tt_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                    else:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                elif clock == 'DunedinPACE':
                    if option.lower() in ordinals:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                else:
                    if option.lower() in ordinals:
                        box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title(
                            'BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p:' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
            if save_figs and num_samples > 15:
                if float(kk_p) < .05 or float(str(tt_p)) < .05:
                    plt.savefig('significant association figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                else:
                    plt.savefig('Non-Significant Association Figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
            plt.close()
s_indices = (set(anti_aging['NONE']) & set(differentials.index))
for option in anti_aging.columns[:]:
    for clock in age_differentials.columns[1:]:
        feat = 'Anti-Aging'
        indices = (set(anti_aging[option].dropna()) & set(differentials.index))
        resp_residuals = differentials.loc[indices, clock]

        if len(resp_residuals.values) > 1:
            standard = differentials.loc[s_indices, clock]

            tt, tt_p = stats.ttest_ind(resp_residuals, standard, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, standard)

            cat_associations.loc['Anti-Aging: ' + option, 'Total Respondents'] = len(resp_residuals)
            cat_associations.loc['Anti-Aging: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            cat_associations.loc['Anti-Aging: ' + option, clock + ' t-test'] = tt
            cat_associations.loc['Anti-Aging: ' + option, clock + ' t-test p-Value'] = tt_p
            cat_associations.loc['Anti-Aging: ' + option, clock + ' Kruskal-Wallis'] = kk
            cat_associations.loc['Anti-Aging: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            excluded = differentials.drop(index=s_indices)[clock]
            num_samples = len(resp_residuals)

            tt, tt_p = stats.ttest_ind(resp_residuals, excluded, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, excluded)

            binary_exclude_associations.loc['Anti-Aging: ' + option, 'Total Respondents'] = len(resp_residuals)
            binary_exclude_associations.loc['Anti-Aging: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            binary_exclude_associations.loc['Anti-Aging: ' + option, clock + ' t-test'] = tt
            binary_exclude_associations.loc['Anti-Aging: ' + option, clock + ' t-test p-Value'] = tt_p
            binary_exclude_associations.loc['Anti-Aging: ' + option, clock + ' Kruskal-Wallis'] = kk
            binary_exclude_associations.loc['Anti-Aging: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            fig = plt.figure()
            # fig.subplots_adjust(bottom=150, top=200)
            box = fig.add_subplot()
            box.boxplot([resp_residuals, excluded])

            avgs = [np.median(x) for x in [resp_residuals, excluded]]
            avg1 = round(np.average(resp_residuals), 2)
            avg2 = round(np.average(excluded), 2)

            indices = [0, 1]
            a, b = np.polyfit(indices, avgs, 1)
            # box.plot(valcopy, a * valcopy + b, 'r-')
            # print(valcopy, a * valcopy + b)

            box.set_xticklabels([option, 'Remainder of cohort'])

            trend = str(round(a, 5))

            if len(option) > 20:
                new_u = option[:20]
            else:
                new_u = option

            plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(resp_residuals)) + ', ' +
                        str(len(excluded)), horizontalalignment='center')

            if clock in immunes:
                if option.lower() in ordinals:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
                else:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
            else:
                if clock == 'TelomerePC':
                    if option.lower() in ordinals:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5)) + ', ' + str(round(tt_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                    else:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                elif clock == 'DunedinPACE':
                    if option.lower() in ordinals:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                else:
                    if option.lower() in ordinals:
                        box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title(
                            'BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p:' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
            if save_figs and num_samples > 15:
                if float(kk_p) < .05 or float(str(tt_p)) < .05:
                    plt.savefig('significant association figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                else:
                    plt.savefig('Non-Significant Association Figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                    '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
            plt.close()

s_indices = (set(supps.columns[0]) & set(differentials.index))
print(supps, supps.columns[0])
for option in supps.columns[1:]:
    for clock in age_differentials.columns[1:]:
        feat = 'Supplements'
        indices = (set(supps[option].dropna()) & set(differentials.index))
        standard = differentials.loc[s_indices, clock]
        resp_residuals = differentials.loc[indices, clock]

        if len(resp_residuals.values) > 1:
            tt, tt_p = stats.ttest_ind(resp_residuals, standard, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, standard)

            cat_associations.loc['Supplements: ' + option, 'Total Respondents'] = len(resp_residuals)
            cat_associations.loc['Supplements: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            cat_associations.loc['Supplements: ' + option, clock + ' t-test'] = tt
            cat_associations.loc['Supplements: ' + option, clock + ' t-test p-Value'] = tt_p
            cat_associations.loc['Supplements: ' + option, clock + ' Kruskal-Wallis'] = kk
            cat_associations.loc['Supplements: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            excluded = differentials.drop(index=s_indices)[clock]

            tt, tt_p = stats.ttest_ind(resp_residuals, excluded, equal_var=False)
            kk, kk_p = stats.kruskal(resp_residuals, excluded)

            binary_exclude_associations.loc['Supplements: ' + option, 'Total Respondents'] = len(resp_residuals)
            binary_exclude_associations.loc['Supplements: ' + option, 'Avg ' + clock] = np.mean(resp_residuals)
            binary_exclude_associations.loc['Supplements: ' + option, clock + ' t-test'] = tt
            binary_exclude_associations.loc['Supplements: ' + option, clock + ' t-test p-Value'] = tt_p
            binary_exclude_associations.loc['Supplements: ' + option, clock + ' Kruskal-Wallis'] = kk
            binary_exclude_associations.loc['Supplements: ' + option, clock + ' Kruskal-Wallis p-Value'] = kk_p

            fig = plt.figure()
            # fig.subplots_adjust(bottom=150, top=200)
            box = fig.add_subplot()
            box.boxplot([resp_residuals, excluded])

            avgs = [np.median(x) for x in [resp_residuals, excluded]]
            avg1 = round(np.average(resp_residuals), 2)
            avg2 = round(np.average(excluded), 2)

            indices = [0, 1]
            a, b = np.polyfit(indices, avgs, 1)
            # box.plot(valcopy, a * valcopy + b, 'r-')
            # print(valcopy, a * valcopy + b)

            box.set_xticklabels([option, 'Remainder of cohort'])

            trend = str(round(a, 5))

            if len(option) > 20:
                new_u = option[:20]
            else:
                new_u = option

            plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(resp_residuals)) + ', ' +
                        str(len(excluded)), horizontalalignment='center')

            if clock in immunes:
                if option.lower() in ordinals:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
                else:
                    box.set_title(
                        'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                            round(kk_p, 5))
                        + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                            avg2))
            else:
                if clock == 'TelomerePC':
                    if option.lower() in ordinals:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5)) + ', ' + str(round(tt_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                    else:
                        box.set_title(
                            'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
                elif clock == 'DunedinPACE':
                    if option.lower() in ordinals:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                else:
                    if option.lower() in ordinals:
                        box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(
                            new_u) + '\nKruskal, t-test p: ' + str(round(kk_p, 5))
                                      + ', ' + str(round(tt_p, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                            avg1) + ', ' + str(avg2))
                    else:
                        box.set_title(
                            'BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p:' + str(
                                round(kk_p, 5))
                            + ', ' + str(round(tt_p, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                avg2))
            if save_figs and num_samples > 15:
                if float(kk_p) < .05 or float(str(tt_p)) < .05:
                    plt.savefig('significant association figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                else:
                    plt.savefig('Non-Significant Association Figures/Binary Comp/' + feat + '-' + str(new_u.replace('/',
                                                                                                                    '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
            plt.close()

if run_categorical:
    for feature in categorical[1:]:
        print('\nCovariate: ', feature)
        featurecopy = feature.lower()

        navals = pd.Series(covariates.loc[:, feature])

        vals = pd.Series(covariates.loc[:, feature]).dropna().astype(str)

        groups = {'GrimAgePC': [],
                  'PhenoAgePC': [],
                  'HorvathPC': [],
                  'HannumPC': [],
                  'TelomerePC': [],
                  'DunedinPACE': [],
                  # 'AltumAge': [],
                  'Full_CumulativeStemCellDivisions': [],
                  'Avg_LifeTime_IntrinsicStemCellDivisionRate': [],
                  'Median_Lifetime_IntrinsicStemCellDivisionRate': [],
                  'Immune.CD8T': [],
                  'Immune.CD4T': [],
                  'Immune.CD4T.CD8T': [],
                  'Immune.NK': [],
                  'Immune.Bcell': [],
                  'Immune.Mono': [],
                  'Immune.Neutrophil': []}

        patients = (set(vals.index) & set(differentials.index))
        diffs = differentials.loc[patients]

        print(feature)
        if featurecopy in ordinals:
            if feature in ['Alcohol Use(times per week)', 'Caffeine Use', 'How often do you use recreational Drugs?']:
                order = ['Never',
                         'On special occasions',
                         'Once per week',
                         '3-5 times per week',
                         'Regularly',
                         ]
            elif feature == 'Tobacco Use':
                order = ['None',
                         'Less than 1 cigarette per week',
                         'Less than 1 cigarette per day',
                         '1-5 cigarettes per day',
                         '6-10 cigarettes per day',
                         '11-20 cigarettes per day',
                         'More than 20 cigarettes per day',
                         ]
            elif feature == 'How often do you exercise per week?':
                order = ['Never',
                         '1-2 times per week',
                         '3-4 times per week',
                         '5-7 times per week',
                         '8 or more times per week'
                         ]
            elif feature == 'Sexual Frequency?':
                order = ['Inactive',
                         'Occasionally',
                         'Regular',
                         ]
            elif feature == 'Hours of sleep per night?':
                order = ['5 hours or less',
                         '6 hours or less',
                         '6 to 8 hours',
                         'More than 8 hours'
                         ]
            elif feature in ['Level of Education', 'Education of Mother', 'Education of Father']:
                order = ['Did not complete high school',
                         'High school or equivalent',
                         'Some college coursework completed',
                         'Associate degree',
                         'Technical or occupational certificate',
                         'Bachelor’s degree',
                         'Master’s degree',
                         'Doctorate (PhD)',
                         'Professional (MD, DO, DDS, JD)',
                         ]
            elif feature == 'Menopause':
                order = ['No', 'Yes']
            for o in range(len(order)):
                order[o] = order[o].lower()
            orderNum = {key: i for i, key in enumerate(order)}

            # is_yes = covariates[covariates["Menopause"]=='Yes'].index
            # is_yes = (set(differentials.index) & set(is_yes))
            # yesDiff = differentials.loc[is_yes, "GrimAgePC"]
            # noDiff = differentials.drop(index=is_yes)['GrimAgePC']

            vals = pd.DataFrame(vals)
            vals[feature] = pd.Categorical(vals[feature], order)
            vals.sort_values(feature, inplace=True)
            vals = pd.Series(vals[feature]).astype(str)

        if feature == 'Alcohol Use(times per week)':
            standard = 'never'
        elif feature == 'Biological Sex':
            standard = 'male'
        elif feature == 'Blood Type':
            standard = 'o positive'
        elif feature == 'Level of Education':
            standard = 'did not complete high school'
        elif feature == 'Education of Mother':
            standard = 'did not complete high school'
        elif feature == 'Education of Father':
            standard = 'did not complete high school'
        elif feature == 'Caffeine Use':
            standard = 'never'
        elif feature == 'Tobacco Use':
            standard = 'none'
        elif feature == 'Recreational Drug Use?':
            standard = 'none'
        elif feature == 'How often do you use recreational Drugs?':
            standard = 'never'
        elif feature == 'How often do you exercise per week?':
            standard = 'never'
        elif feature == 'Exercise Type?':
            standard = 'strength'
        elif feature == 'Sexual Frequency?':
            standard = 'inactive'
        elif feature == 'Hours of sleep per night?':
            standard = '6 to 8 hours'
        elif feature == 'Menopause':
            standard = 'no'
        elif feature == 'Given Birth':
            standard = 'no'
        elif feature == 'Mother Nicotine Use':
            standard = 'no'
        elif feature == 'Mother Pregnancy Complications':
            standard = 'no'
        elif feature == 'Diagnosed with low or high calcium':
            standard = 'no'
        elif feature == 'Marital Status':
            standard = 'single'
        elif feature == 'Caffeine Use':
            standard = 'never'
        elif feature == 'Tobacco Use':
            standard = 'none'
        elif feature == 'Ethnicity':
            # standard = 'european or caucasian;'
            standard = 'asian;'
        elif feature == 'What does your diet mostly consist of?':
            standard = "balanced (mixed; red/white meat, veg, carb, etc;'regular')"
        elif feature in n_list:
            standard = 'none'
        else:
            standard = '0.0'
            print('Non-matching covariate and logic, rewrite missing covariate: ', feature)

        vals = vals.loc[patients]
        standard_patients = vals[vals == standard].index  # standard].index
        vals.replace(' ', '', inplace=True)

        if feature.lower() in ['given birth', 'menopause']:
            is_male = covariates[covariates['Biological Sex'] == 'male'].index
            to_drop = (set(is_male) & set(vals.index))
            vals.drop(index=to_drop, inplace=True)
            patients = [x for x in patients if x not in to_drop]

        uniques = np.unique(vals)
        valcopy = vals.copy()

        for v in range(len(uniques)):
            value = uniques[v]
            valcopy.replace(value, v, inplace=True)

        GrimTT = {}
        PhenoTT = {}
        HorPC_TT = {}
        HanPC_TT = {}
        teloPC_TT = {}
        duneTT = {}
        # altumTT = {}
        fullCellTT = {}
        avgCellTT = {}
        medianCellTT = {}
        cd8TT = {}
        cd4TT = {}
        cdRatioTT = {}
        nkTT = {}
        bCellTT = {}
        monoTT = {}
        neutroTT = {}

        GrimTT_P = {}
        PhenoTT_P = {}
        HorPC_TT_P = {}
        HanPC_TT_P = {}
        teloPC_TT_P = {}
        duneTT_P = {}
        # altumTT_P = {}
        fullCellTT_P = {}
        avgCellTT_P = {}
        medianCellTT_P = {}
        cd8TT_P = {}
        cd4TT_P = {}
        cdRatioTT_P = {}
        nkTT_P = {}
        bCellTT_P = {}
        monoTT_P = {}
        neutroTT_P = {}

        groups = {'GrimAgePC': [],
                  'PhenoAgePC': [],
                  'HorvathPC': [],
                  'HannumPC': [],
                  'TelomerePC': [],
                  'DunedinPACE': [],
                  # 'AltumAge': [],
                  'Full_CumulativeStemCellDivisions': [],
                  'Avg_LifeTime_IntrinsicStemCellDivisionRate': [],
                  'Median_Lifetime_IntrinsicStemCellDivisionRate': [],
                  'Immune.CD8T': [],
                  'Immune.CD4T': [],
                  'Immune.CD4T.CD8T': [],
                  'Immune.NK': [],
                  'Immune.Bcell': [],
                  'Immune.Mono': [],
                  'Immune.Neutrophil': [],
                  }

        if feature.lower() == 'do you take any of the following nutritional supplements?':
            feat = 'Nutritional Supplements'
        elif feature.lower() == 'do you take any of the following supplements or medications?':
            feat = 'Supplements'
        elif feature.lower() == 'recreational drug use?':
            feat = 'Drugs'
        elif feature.lower() == 'what does your diet mostly consist of?':
            feat = 'Diet'
        elif feature.lower() == 'how often do you use recreational drugs?':
            feat = 'DrugFrequency'
        elif feature.lower() == 'how often do you exercise per week?':
            feat = 'ExerciseFrequency'
        elif feature.lower() == 'actively engage in anti-aging interventions?':
            feat = 'Anti-aging interventions'
        elif feature.lower() == 'alcohol use(times per week)':
            feat = 'alcohol use'
        else:
            feat = feature.replace('?', '').replace('/', '-')

        demo_residuals = {}
        for u in np.unique(vals):
            unique_patients = vals[vals == u].index
            unique_cats = vals.loc[unique_patients]

            def appending(c):
                unique_residuals = diffs.loc[unique_patients, c]
                standard_residuals = diffs.loc[standard_patients, c]

                if list(unique_residuals) == list(standard_residuals):
                    pass
                else:
                    if c == 'GrimAgePC':
                        var_name = 'Grim'
                    elif c == 'PhenoAgePC':
                        var_name = 'Pheno'
                    elif c == 'HorvathPC':
                        var_name = 'HorPC_'
                    elif c == 'HannumPC':
                        var_name = 'HanPC_'
                    elif c == 'TelomerePC':
                        var_name = 'teloPC_'
                    elif c == 'DunedinPACE':
                        var_name = 'dune'
                    elif c == 'AltumAge':
                        var_name = 'altum'
                    elif clock == 'Full_CumulativeStemCellDivisions':
                        var_name = 'fullCell'
                    elif clock == 'Avg_LifeTime_IntrinsicStemCellDivisionRate':
                        var_name = 'avgCell'
                    elif clock == 'Median_Lifetime_IntrinsicStemCellDivisionRate':
                        var_name = 'medianCell'
                    elif c == 'Immune.CD8T':
                        var_name = 'cd8'
                    elif c == 'Immune.CD4T':
                        var_name = 'cd4'
                    elif c == 'Immune.CD4T.CD8T':
                        var_name = 'cdRatio'
                    elif c == 'Immune.NK':
                        var_name = 'nk'
                    elif c == 'Immune.Bcell':
                        var_name = 'bCell'
                    elif c == 'Immune.Mono':
                        var_name = 'mono'
                    elif c == 'Immune.Neutrophil':
                        var_name = 'neutro'

                    num_samples = len(unique_residuals)
                    if num_samples > 1:
                        tt, ttp = scipy.stats.ttest_ind(unique_residuals, standard_residuals, equal_var=False)
                        try:
                            kk, kkp = stats.kruskal(unique_residuals, standard_residuals)
                        except:
                            kk, kkp = np.nan, np.nan

                        exec(var_name + 'TT[u] = [tt, num_samples]')
                        exec(var_name + 'TT_P[u] = [ttp, num_samples]')

                        fig = plt.figure()
                        # fig.subplots_adjust(bottom=150, top=200)
                        box = fig.add_subplot()
                        box.boxplot([unique_residuals, standard_residuals])

                        avgs = [np.median(x) for x in [unique_residuals, standard_residuals]]
                        avg1 = round(np.average(unique_residuals), 2)
                        avg2 = round(np.average(standard_residuals), 2)

                        indices = [0, 1]
                        a, b = np.polyfit(indices, avgs, 1)
                        # box.plot(valcopy, a * valcopy + b, 'r-')
                        # print(valcopy, a * valcopy + b)

                        box.set_xticklabels([u, standard])

                        trend = str(round(a, 5))

                        if len(u) > 20:
                            new_u = u[:10] + u[-10:]
                        else:
                            new_u = u

                        plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(unique_residuals)) + ', ' +
                                    str(len(standard_residuals)), horizontalalignment='center')

                        if clock in immunes:
                            if featurecopy in ordinals:
                                box.set_title(
                                    'Standard cell values for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                    + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                                        avg1) + ', ' + str(avg2))
                            else:
                                box.set_title('Standard cell values for ' + clock + ', ' + feat + '-' + str(
                                    new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                                    avg1) + ', ' + str(avg2))
                        else:
                            if clock == 'TelomerePC':
                                if featurecopy in ordinals:
                                    box.set_title(
                                        'Standard lengths for ' + clock + ', ' + feat + '-' + str(
                                            new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5)) + ', ' + str(
                                            round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(
                                            avg1) + ', ' + str(avg2))
                                else:
                                    box.set_title('Standard lengths for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                                  + ', ' + str(
                                        round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                        avg2))
                            elif clock == 'DunedinPACE':
                                if featurecopy in ordinals:
                                    box.set_title('Standard aging Rate for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                                  + ', ' + str(
                                        round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                        avg2))
                                else:
                                    box.set_title('Standard aging Rate for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                                  + ', ' + str(
                                        round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                        avg2))
                            else:
                                if featurecopy in ordinals:
                                    box.set_title('Standard residuals for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                                  + ', ' + str(
                                        round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(
                                        avg2))
                                else:
                                    box.set_title('Standard residuals for ' + clock + ', ' + feat + '-' + str(
                                        new_u) + '\nKruskal, t-test p:' + str(round(kkp, 5))
                                                  + ', ' + str(round(ttp, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(
                                        avg1) + ', ' + str(avg2))
                        if save_figs and num_samples > 15:
                                if float(kkp) < .05 or float(str(ttp)) < .05:
                                    plt.savefig('significant association figures/Standard Comp/' + feat + '-' + str(
                                        new_u.replace('/',  '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                                else:
                                    plt.savefig('Non-Significant Association Figures/Standard Comp/' + feat + '-' + str(
                                        new_u.replace('/', '-')) + '-' + clock + '.png')  # + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                            # else:
                        # if show_residuals:
                        #     plt.show()

                        plt.close()

            each_clock = {}
            for clock in groups.keys():
                appending(c=clock)

                unique_residuals = diffs.loc[unique_patients, clock]
                standard_residuals = diffs.loc[standard_patients, clock]
                binary_excluded_residuals = diffs.drop(index=unique_patients)[clock]

                each_clock[clock] = unique_residuals

                num_samples = len(unique_residuals)
                if num_samples > 1:
                    if list(unique_residuals) == list(standard_residuals):
                        pass
                    else:
                        tt, ttp = scipy.stats.ttest_ind(unique_residuals, standard_residuals, equal_var=False)
                        try:
                            kk, kkp = stats.kruskal(unique_residuals, standard_residuals)
                        except:
                            kk, kkp = np.nan, np.nan

                        cat_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(unique_residuals)
                        cat_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(unique_residuals)
                        cat_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
                        cat_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
                        cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
                        cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp

                    tt, ttp = scipy.stats.ttest_ind(unique_residuals, binary_excluded_residuals, equal_var=False)
                    try:
                        kk, kkp = stats.kruskal(unique_residuals, binary_excluded_residuals)
                    except:
                        kk, kkp = np.nan, np.nan

                    binary_exclude_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(unique_residuals)
                    binary_exclude_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(unique_residuals)
                    binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
                    binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
                    binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
                    binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp

                    fig = plt.figure()
                    # fig.subplots_adjust(bottom=150, top=200)
                    box = fig.add_subplot()
                    box.boxplot([unique_residuals, binary_excluded_residuals])

                    avgs = [np.median(x) for x in [unique_residuals, binary_excluded_residuals]]
                    avg1 = round(np.average(unique_residuals), 2)
                    avg2 = round(np.average(binary_excluded_residuals), 2)

                    indices = [0, 1]
                    a, b = np.polyfit(indices, avgs, 1)
                    # box.plot(valcopy, a * valcopy + b, 'r-')
                    # print(valcopy, a * valcopy + b)

                    box.set_xticklabels([u, 'Remainder of cohort'])

                    trend = str(round(a, 5))

                    if len(u) > 20:
                        new_u = u[:20]
                    else:
                        new_u = u

                    plt.figtext(0.5, 0.01, 'Sample sizes: ' + str(len(unique_residuals)) + ', ' +
                                str(len(binary_excluded_residuals)), horizontalalignment='center')

                    if clock in immunes:
                        if featurecopy in ordinals:
                            box.set_title(
                                'BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                        else:
                            box.set_title('BC cell values for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                          + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                    else:
                        if clock == 'TelomerePC':
                            if featurecopy in ordinals:
                                box.set_title(
                                    'BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5)) + ', ' + str(round(ttp, 5))
                                + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                            else:
                                box.set_title('BC lengths for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                        elif clock == 'DunedinPACE':
                            if featurecopy in ordinals:
                                box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                            else:
                                box.set_title('BC aging Rate for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                        else:
                            if featurecopy in ordinals:
                                box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p: ' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend: ' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                            else:
                                box.set_title('BC residuals for ' + clock + ', ' + feat + '-' + str(new_u) + '\nKruskal, t-test p:' + str(round(kkp, 5))
                                              + ', ' + str(round(ttp, 5)) + ' | Trend:' + trend + ' | Avgs: ' + str(avg1) + ', ' + str(avg2))
                    if save_figs and num_samples > 15:
                        if float(kkp) < .05 or float(str(ttp)) < .05:
                            plt.savefig('significant association figures/Binary Comp/' + feat + '-' + str(new_u.replace('/', '-')) + '-' + clock + '.png')  #  + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                        else:
                            plt.savefig('Non-Significant Association Figures/Binary Comp/' + feat + '-' + str(new_u.replace('/', '-')) + '-' + clock + '.png')  #  + '-kk_p-' + str(round(kkp, 5)) + 'T-Test-' + str(round(ttp, 5)) + '.png')
                        # else:
                    # if show_residuals:
                    #     plt.show()

                    plt.close()

                    # if kkp < .05 or ttp < .05:
                    #     demo_residuals[u] = each_clock

                    # for clock in age_differentials.columns[1:]:
                    #     try:
                    #         labels = demo_residuals.keys()
                    #         fig = plt.figure()
                    #         box = fig.add_subplot()
                    #         # fig.subplots_adjust(bottom=150, top=200)
                    #         box.boxplot([demo_residuals[k][clock] for k in demo_residuals.keys()])
                    #
                    #         if featurecopy in ordinals:
                    #             avgs = [np.median(x) for x in [demo_residuals[k][clock] for k in demo_residuals.keys()]]
                    #
                    #             indices = list(range(len(demo_residuals.keys())))
                    #             indices = [x+1 for x in indices]
                    #             a, b = np.polyfit(indices, avgs, 1)
                    #             box.plot(valcopy, a * valcopy + b, 'r-')
                    #
                    #         box.set_xticklabels(demo_residuals.keys())
                    #
                    #         if featurecopy in ordinals:
                    #             trend = str(round(a, 5))
                    #
                    #         if clock in immunes:
                    #             if featurecopy in ordinals:
                    #                 box.set_title('Significant cell values for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + ' || T-Test p-Val: ' + ' || Trend: ' + trend)
                    #             else:
                    #                 box.set_title('Significant cell values for ' + clock + ', ' + feat + '\nKruskal p-Val: ')
                    #         else:
                    #             if clock == 'TelomerePC':
                    #                 if featurecopy in ordinals:
                    #                     box.set_title('Significant lengths for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + ' || T-Test p-Val: ' + ' || Trend: ' + trend)
                    #                 else:
                    #                     box.set_title('Significant lengths for ' + clock + ', ' + feat + '\nKruskal p-Val: ')
                    #             elif clock == 'DunedinPACE':
                    #                 if featurecopy in ordinals:
                    #                     box.set_title('Significant aging Rate for ' + clock + ', ' + feat + '\nKruskal p-Val: '
                    #                                   + ' || T-Test p-Val: ' + ' || Trend: ' + trend)
                    #                 else:
                    #                     box.set_title('Significant aging Rate for ' + clock + ', ' + feat + '\nKruskal p-Val: ')
                    #             else:
                    #                 if featurecopy in ordinals:
                    #                     box.set_title('Significant residuals for ' + clock + ', ' + feat + '\nKruskal p-Val: '
                    #                                   + ' || T-Test p-Val: ' + ' || Trend: ' + trend)
                    #                 else:
                    #                     box.set_title('Significant residuals for ' + clock + ', ' + feat + '\nKruskal p-Val: ')
                    #         if save_figs:
                    #         #     if float(be_kkp) < .05 or float(str(be_ttp)) < .05:
                    #             plt.savefig('significant association figures/' + feat + '-' + clock + 'kk_p-' + 'T-Test-' + '.png')
                    #             # else:
                    #         if show_residuals:
                    #             plt.show()
                    #
                    #         plt.close()
                    #     except Exception as e:
                    #         print('Error in significant graphing... ', e)

        if featurecopy in ordinals:
            uniques = order
        else:
            uniques = np.unique(vals)

        for u in uniques:
            unique_patients = vals[vals == u].index
            for clock in groups.keys():
                groups[clock].append(list(diffs.loc[unique_patients, clock + '']))

        GrimANOVA, GrimANOVA_P = anova(groups_='groups', length=len(groups['GrimAgePC']), clock_='GrimAgePC')
        PhenoANOVA, PhenoANOVA_P = anova(groups_='groups', length=len(groups['PhenoAgePC']), clock_='PhenoAgePC')
        HorPC_ANOVA, HorPC_ANOVA_P = anova(groups_='groups', length=len(groups['HorvathPC']), clock_='HorvathPC')
        HanPC_ANOVA, HanPC_ANOVA_P = anova(groups_='groups', length=len(groups['HannumPC']), clock_='HannumPC')
        teloPC_ANOVA, teloPC_ANOVA_P = anova(groups_='groups', length=len(groups['TelomerePC']), clock_='TelomerePC')
        duneANOVA, duneANOVA_P = anova(groups_='groups', length=len(groups['DunedinPACE']), clock_='DunedinPACE')
        # altumANOVA, altumANOVA_P = anova(groups_='groups', length=len(groups['AltumAge']), clock_='AltumAge')
        fullCellANOVA, fullCellANOVA_P = anova(groups_='groups', length=len(groups['Full_CumulativeStemCellDivisions']), clock_='Full_CumulativeStemCellDivisions')
        avgCellANOVA, avgCellANOVA_P = anova(groups_='groups', length=len(groups['Avg_LifeTime_IntrinsicStemCellDivisionRate']), clock_='Avg_LifeTime_IntrinsicStemCellDivisionRate')
        medianCellANOVA, medianCellANOVA_P = anova(groups_='groups', length=len(groups['Median_Lifetime_IntrinsicStemCellDivisionRate']), clock_='Median_Lifetime_IntrinsicStemCellDivisionRate')
        cd8ANOVA, cd8ANOVA_P = anova(groups_='groups', length=len(groups['Immune.CD8T']), clock_='Immune.CD8T')
        cd4ANOVA, cd4ANOVA_P = anova(groups_='groups', length=len(groups['Immune.CD4T']), clock_='Immune.CD4T')
        cdRatioANOVA, cdRatioANOVA_P = anova(groups_='groups', length=len(groups['Immune.CD4T.CD8T']), clock_='Immune.CD4T.CD8T')
        nkANOVA, nkANOVA_P = anova(groups_='groups', length=len(groups['Immune.NK']), clock_='Immune.NK')
        bCellANOVA, bCellANOVA_P = anova(groups_='groups', length=len(groups['Immune.Bcell']), clock_='Immune.Bcell')
        monoANOVA, monoANOVA_P = anova(groups_='groups', length=len(groups['Immune.Mono']), clock_='Immune.Mono')
        neutroANOVA, neutroANOVA_P = anova(groups_='groups', length=len(groups['Immune.Neutrophil']), clock_='Immune.Neutrophil')

        GrimKK, GrimKK_P = kruskal(groups_='groups', length=len(groups['GrimAgePC']), clock_='GrimAgePC')
        PhenoKK, PhenoKK_P = kruskal(groups_='groups', length=len(groups['PhenoAgePC']), clock_='PhenoAgePC')
        HorPC_KK, HorPC_KK_P = kruskal(groups_='groups', length=len(groups['HorvathPC']), clock_='HorvathPC')
        HanPC_KK, HanPC_KK_P = kruskal(groups_='groups', length=len(groups['HannumPC']), clock_='HannumPC')
        teloPC_KK, teloPC_KK_P = kruskal(groups_='groups', length=len(groups['TelomerePC']), clock_='TelomerePC')
        duneKK, duneKK_P = kruskal(groups_='groups', length=len(groups['DunedinPACE']), clock_='DunedinPACE')
        # altumKK, altumKK_P = kruskal(groups_='groups', length=len(groups['AltumAge']), clock_='AltumAge')
        fullCellKK, fullCellKK_P = kruskal(groups_='groups', length=len(groups['Full_CumulativeStemCellDivisions']), clock_='Full_CumulativeStemCellDivisions')
        avgCellKK, avgCellKK_P = kruskal(groups_='groups', length=len(groups['Avg_LifeTime_IntrinsicStemCellDivisionRate']), clock_='Avg_LifeTime_IntrinsicStemCellDivisionRate')
        medianCellKK, medianCellKK_P = kruskal(groups_='groups', length=len(groups['Median_Lifetime_IntrinsicStemCellDivisionRate']), clock_='Median_Lifetime_IntrinsicStemCellDivisionRate')
        cd8KK, cd8KK_P = kruskal(groups_='groups', length=len(groups['Immune.CD8T']), clock_='Immune.CD8T')
        cd4KK, cd4KK_P = kruskal(groups_='groups', length=len(groups['Immune.CD4T']), clock_='Immune.CD4T')
        cdRatioKK, cdRatioKK_P = kruskal(groups_='groups', length=len(groups['Immune.CD4T.CD8T']), clock_='Immune.CD4T.CD8T')
        nkKK, nkKK_P = kruskal(groups_='groups', length=len(groups['Immune.NK']), clock_='Immune.NK')
        bCellKK, bCellKK_P = kruskal(groups_='groups', length=len(groups['Immune.Bcell']), clock_='Immune.Bcell')
        monoKK, monoKK_P = kruskal(groups_='groups', length=len(groups['Immune.Mono']), clock_='Immune.Mono')
        neutroKK, neutroKK_P = kruskal(groups_='groups', length=len(groups['Immune.Neutrophil']), clock_='Immune.Neutrophil')

        GrimEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'GrimAgePC'], discrete_features=True)[0]
        PhenoEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'PhenoAgePC'], discrete_features=True)[0]
        HorPC_Entropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'HorvathPC'], discrete_features=True)[0]
        HanPC_Entropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'HannumPC'], discrete_features=True)[0]
        teloPC_Entropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'TelomerePC'], discrete_features=True)[0]
        duneEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'DunedinPACE'], discrete_features=True)[0]
        # altumEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'AltumAge'], discrete_features=True)[0]
        fullCellEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Full_CumulativeStemCellDivisions'], discrete_features=True)[0]
        avgCellEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Avg_LifeTime_IntrinsicStemCellDivisionRate'], discrete_features=True)[0]
        medianCellEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Median_Lifetime_IntrinsicStemCellDivisionRate'], discrete_features=True)[0]
        cd8Entropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.CD8T'], discrete_features=True)[0]
        cd4Entropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.CD4T'], discrete_features=True)[0]
        cdRatioEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.CD4T.CD8T'], discrete_features=True)[0]
        nkEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.NK'], discrete_features=True)[0]
        bCellEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.Bcell'], discrete_features=True)[0]
        monoEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.Mono'], discrete_features=True)[0]
        neutroEntropy = mutual_info_regression(np.array(valcopy).reshape(-1, 1), diffs.loc[patients, 'Immune.Neutrophil'], discrete_features=True)[0]

        associations.at[feat, 'GrimAgePC T-Test'] = np.array(GrimTT, dtype=object)
        associations.at[feat, 'GrimAgePC T-Test p-Value'] = np.array(GrimTT_P, dtype=object)
        associations.loc[feat, 'GrimAgePC ANOVA'] = GrimANOVA
        associations.loc[feat, 'GrimAgePC ANOVA p-Value'] = GrimANOVA_P
        associations.loc[feat, 'GrimAgePC Kruskal-Wallis'] = GrimKK
        associations.loc[feat, 'GrimAgePC Kruskal-Wallis p-Value'] = GrimKK_P
        associations.loc[feat, 'GrimAgePC Entropy/Importance Categorical'] = GrimEntropy

        associations.at[feat, 'PhenoAgePC T-Test'] = np.array(PhenoTT, dtype=object)
        associations.at[feat, 'PhenoAgePC T-Test p-Value'] = np.array(PhenoTT_P, dtype=object)
        associations.loc[feat, 'PhenoAgePC ANOVA'] = PhenoANOVA
        associations.loc[feat, 'PhenoAgePC ANOVA p-Value'] = PhenoANOVA_P
        associations.loc[feat, 'PhenoAgePC Kruskal-Wallis'] = PhenoKK
        associations.loc[feat, 'PhenoAgePC Kruskal-Wallis p-Value'] = PhenoKK_P
        associations.loc[feat, 'PhenoAgePC Entropy/Importance Categorical'] = PhenoEntropy

        associations.at[feat, 'HorvathPC T-Test'] = np.array(HorPC_TT, dtype=object)
        associations.at[feat, 'HorvathPC T-Test p-Value'] = np.array(HorPC_TT_P, dtype=object)
        associations.loc[feat, 'HorvathPC ANOVA'] = HorPC_ANOVA
        associations.loc[feat, 'HorvathPC ANOVA p-Value'] = HorPC_ANOVA_P
        associations.loc[feat, 'HorvathPC Kruskal-Wallis'] = HorPC_KK
        associations.loc[feat, 'HorvathPC Kruskal-Wallis p-Value'] = HorPC_KK_P
        associations.loc[feat, 'HorvathPC Entropy/Importance Categorical'] = HorPC_Entropy

        associations.at[feat, 'HannumPC T-Test'] = np.array(HanPC_TT, dtype=object)
        associations.at[feat, 'HannumPC T-Test p-Value'] = np.array(HanPC_TT_P, dtype=object)
        associations.loc[feat, 'HannumPC ANOVA'] = HanPC_ANOVA
        associations.loc[feat, 'HannumPC ANOVA p-Value'] = HanPC_ANOVA_P
        associations.loc[feat, 'HannumPC Kruskal-Wallis'] = HanPC_KK
        associations.loc[feat, 'HannumPC Kruskal-Wallis p-Value'] = HanPC_KK_P
        associations.loc[feat, 'HannumPC Entropy/Importance Categorical'] = HanPC_Entropy

        associations.at[feat, 'TelomerePC T-Test'] = np.array(teloPC_TT, dtype=object)
        associations.at[feat, 'TelomerePC T-Test p-Value'] = np.array(teloPC_TT_P, dtype=object)
        associations.loc[feat, 'TelomerePC ANOVA'] = teloPC_ANOVA
        associations.loc[feat, 'TelomerePC ANOVA p-Value'] = teloPC_ANOVA_P
        associations.loc[feat, 'TelomerePC Kruskal-Wallis'] = teloPC_KK
        associations.loc[feat, 'TelomerePC Kruskal-Wallis p-Value'] = teloPC_KK_P
        associations.loc[feat, 'TelomerePC Entropy/Importance Categorical'] = teloPC_Entropy

        associations.at[feat, 'DunedinPACE T-Test'] = np.array(duneTT, dtype=object)
        associations.at[feat, 'DunedinPACE T-Test p-Value'] = np.array(duneTT_P, dtype=object)
        associations.loc[feat, 'DunedinPACE ANOVA'] = duneANOVA
        associations.loc[feat, 'DunedinPACE ANOVA p-Value'] = duneANOVA_P
        associations.loc[feat, 'DunedinPACE Kruskal-Wallis'] = duneKK
        associations.loc[feat, 'DunedinPACE Kruskal-Wallis p-Value'] = duneKK_P
        associations.loc[feat, 'DunedinPACE Entropy/Importance Categorical'] = duneEntropy

        # associations.at[feat, 'AltumAge T-Test'] = np.array(altumTT, dtype=object)
        # associations.at[feat, 'AltumAge T-Test p-Value'] = np.array(altumTT_P, dtype=object)
        # associations.loc[feat, 'AltumAge ANOVA'] = altumANOVA
        # associations.loc[feat, 'AltumAge ANOVA p-Value'] = altumANOVA_P
        # associations.loc[feat, 'AltumAge Kruskal-Wallis'] = altumKK
        # associations.loc[feat, 'AltumAge Kruskal-Wallis p-Value'] = altumKK_P
        # associations.loc[feat, 'AltumAge Entropy/Importance Categorical'] = altumEntropy

        associations.at[feat, 'Full_CumulativeStemCellDivisions T-Test'] = np.array(fullCellTT, dtype=object)
        associations.at[feat, 'Full_CumulativeStemCellDivisions T-Test p-Value'] = np.array(fullCellTT_P, dtype=object)
        associations.loc[feat, 'Full_CumulativeStemCellDivisions ANOVA'] = fullCellANOVA
        associations.loc[feat, 'Full_CumulativeStemCellDivisions ANOVA p-Value'] = fullCellANOVA_P
        associations.loc[feat, 'Full_CumulativeStemCellDivisions Kruskal-Wallis'] = fullCellKK
        associations.loc[feat, 'Full_CumulativeStemCellDivisions Kruskal-Wallis p-Value'] = fullCellKK_P
        associations.loc[feat, 'Full_CumulativeStemCellDivisions Entropy/Importance Categorical'] = fullCellEntropy

        associations.at[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate T-Test'] = np.array(avgCellTT, dtype=object)
        associations.at[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate T-Test p-Value'] = np.array(avgCellTT_P, dtype=object)
        associations.loc[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate ANOVA'] = avgCellANOVA
        associations.loc[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate ANOVA p-Value'] = avgCellANOVA_P
        associations.loc[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Kruskal-Wallis'] = avgCellKK
        associations.loc[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Kruskal-Wallis p-Value'] = avgCellKK_P
        associations.loc[feat, 'Avg_LifeTime_IntrinsicStemCellDivisionRate Entropy/Importance Categorical'] = avgCellEntropy

        associations.at[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate T-Test'] = np.array(medianCellTT, dtype=object)
        associations.at[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate T-Test p-Value'] = np.array(medianCellTT_P, dtype=object)
        associations.loc[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate ANOVA'] = medianCellANOVA
        associations.loc[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate ANOVA p-Value'] = medianCellANOVA_P
        associations.loc[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate Kruskal-Wallis'] = medianCellKK
        associations.loc[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate Kruskal-Wallis p-Value'] = medianCellKK_P
        associations.loc[feat, 'Median_Lifetime_IntrinsicStemCellDivisionRate Entropy/Importance Categorical'] = medianCellEntropy

        associations.at[feat, 'Immune.CD8T T-Test'] = np.array(cd8TT, dtype=object)
        associations.at[feat, 'Immune.CD8T T-Test p-Value'] = np.array(cd8TT_P, dtype=object)
        associations.loc[feat, 'Immune.CD8T ANOVA'] = cd8ANOVA
        associations.loc[feat, 'Immune.CD8T ANOVA p-Value'] = cd8ANOVA_P
        associations.loc[feat, 'Immune.CD8T Kruskal-Wallis'] = cd8KK
        associations.loc[feat, 'Immune.CD8T Kruskal-Wallis p-Value'] = cd8KK_P
        associations.loc[feat, 'Immune.CD8T Entropy/Importance Categorical'] = cd8Entropy

        associations.at[feat, 'Immune.CD4T T-Test'] = np.array(cd4TT, dtype=object)
        associations.at[feat, 'Immune.CD4T T-Test p-Value'] = np.array(cd4TT_P, dtype=object)
        associations.loc[feat, 'Immune.CD4T ANOVA'] = cd4ANOVA
        associations.loc[feat, 'Immune.CD4T ANOVA p-Value'] = cd4ANOVA_P
        associations.loc[feat, 'Immune.CD4T Kruskal-Wallis'] = cd4KK
        associations.loc[feat, 'Immune.CD4T Kruskal-Wallis p-Value'] = cd4KK_P
        associations.loc[feat, 'Immune.CD4T Entropy/Importance Categorical'] = cd4Entropy

        associations.at[feat, 'Immune.CD4T.CD8T T-Test'] = np.array(cdRatioTT, dtype=object)
        associations.at[feat, 'Immune.CD4T.CD8T T-Test p-Value'] = np.array(cdRatioTT_P, dtype=object)
        associations.loc[feat, 'Immune.CD4T.CD8T ANOVA'] = cdRatioANOVA
        associations.loc[feat, 'Immune.CD4T.CD8T ANOVA p-Value'] = cdRatioANOVA_P
        associations.loc[feat, 'Immune.CD4T.CD8T Kruskal-Wallis'] = cdRatioKK
        associations.loc[feat, 'Immune.CD4T.CD8T Kruskal-Wallis p-Value'] = cdRatioKK_P
        associations.loc[feat, 'Immune.CD4T.CD8T Entropy/Importance Categorical'] = cdRatioEntropy

        associations.at[feat, 'Immune.NK T-Test'] = np.array(nkTT, dtype=object)
        associations.at[feat, 'Immune.NK T-Test p-Value'] = np.array(nkTT_P, dtype=object)
        associations.loc[feat, 'Immune.NK ANOVA'] = nkANOVA
        associations.loc[feat, 'Immune.NK ANOVA p-Value'] = nkANOVA_P
        associations.loc[feat, 'Immune.NK Kruskal-Wallis'] = nkKK
        associations.loc[feat, 'Immune.NK Kruskal-Wallis p-Value'] = nkKK_P
        associations.loc[feat, 'Immune.NK Entropy/Importance Categorical'] = nkEntropy

        associations.at[feat, 'Immune.Bcell T-Test'] = np.array(bCellTT, dtype=object)
        associations.at[feat, 'Immune.Bcell T-Test p-Value'] = np.array(bCellTT_P, dtype=object)
        associations.loc[feat, 'Immune.Bcell ANOVA'] = bCellANOVA
        associations.loc[feat, 'Immune.Bcell ANOVA p-Value'] = bCellANOVA_P
        associations.loc[feat, 'Immune.Bcell Kruskal-Wallis'] = bCellKK
        associations.loc[feat, 'Immune.Bcell Kruskal-Wallis p-Value'] = bCellKK_P
        associations.loc[feat, 'Immune.Bcell Entropy/Importance Categorical'] = bCellEntropy

        associations.at[feat, 'Immune.Mono T-Test'] = np.array(monoTT, dtype=object)
        associations.at[feat, 'Immune.Mono T-Test p-Value'] = np.array(monoTT_P, dtype=object)
        associations.loc[feat, 'Immune.Mono ANOVA'] = monoANOVA
        associations.loc[feat, 'Immune.Mono ANOVA p-Value'] = monoANOVA_P
        associations.loc[feat, 'Immune.Mono Kruskal-Wallis'] = monoKK
        associations.loc[feat, 'Immune.Mono Kruskal-Wallis p-Value'] = monoKK_P
        associations.loc[feat, 'Immune.Mono Entropy/Importance Categorical'] = monoEntropy

        associations.at[feat, 'Immune.Neutrophil T-Test'] = np.array(neutroTT, dtype=object)
        associations.at[feat, 'Immune.Neutrophil T-Test p-Value'] = np.array(neutroTT_P, dtype=object)
        associations.loc[feat, 'Immune.Neutrophil ANOVA'] = neutroANOVA
        associations.loc[feat, 'Immune.Neutrophil ANOVA p-Value'] = neutroANOVA_P
        associations.loc[feat, 'Immune.Neutrophil Kruskal-Wallis'] = neutroKK
        associations.loc[feat, 'Immune.Neutrophil Kruskal-Wallis p-Value'] = neutroKK_P
        associations.loc[feat, 'Immune.Neutrophil Entropy/Importance Categorical'] = neutroEntropy

        demographics = {}

        for u in uniques:
            total_num = list(vals).count(u)
            demographics[u] = total_num
            print(u, ': ', total_num)
        plt.bar(demographics.keys(), demographics.values())
        plt.title(feat)
        plt.xlabel(demographics.keys())

        if save_figs:
            plt.savefig('Demographics/' + feat + '.png')
        if show_demographics:
            plt.show()
        plt.close()

        for clock in age_differentials.columns[1:]:
            demo_residuals = {}
            for u in uniques:
                unique_pats = vals[vals == u].index
                demo_residuals[u] = diffs.loc[unique_pats, clock]

            demo_split(feat, feature, vals, clock)

            fig = plt.figure()
            # fig.subplots_adjust(bottom=150, top=200)
            box = fig.add_subplot()
            box.boxplot(demo_residuals.values())

            if featurecopy in ordinals:
                avgs = [np.median(x) for x in demo_residuals.values()]

                indices = list(range(len(demo_residuals.keys())))
                indices = [x+1 for x in indices]
                a, b = np.polyfit(indices, avgs, 1)
                box.plot(valcopy, a * valcopy + b, 'r-')

            box.set_xticklabels(demo_residuals.keys())

            if clock == 'GrimAgePC':
                var_name = 'Grim'
            elif clock == 'PhenoAgePC':
                var_name = 'Pheno'
            elif clock == 'HorvathPC':
                var_name = 'HorPC_'
            elif clock == 'HannumPC':
                var_name = 'HanPC_'
            elif clock == 'TelomerePC':
                var_name = 'teloPC_'
            elif clock == 'DunedinPACE':
                var_name = 'dune'
            elif clock == 'AltumAge':
                var_name = 'altum'
            elif clock == 'Full_CumulativeStemCellDivisions':
                var_name = 'fullCell'
            elif clock == 'Avg_LifeTime_IntrinsicStemCellDivisionRate':
                var_name = 'avgCell'
            elif clock == 'Median_Lifetime_IntrinsicStemCellDivisionRate':
                var_name = 'medianCell'
            elif clock == 'Immune.CD8T':
                var_name = 'cd8'
            elif clock == 'Immune.CD4T':
                var_name = 'cd4'
            elif clock == 'Immune.CD4T.CD8T':
                var_name = 'cdRatio'
            elif clock == 'Immune.NK':
                var_name = 'nk'
            elif clock == 'Immune.Bcell':
                var_name = 'bCell'
            elif clock == 'Immune.Mono':
                var_name = 'mono'
            elif clock == 'Immune.Neutrophil':
                var_name = 'neutro'

            eval_string = var_name + 'KK_P'
            eval_stringANOVA = var_name + 'ANOVA_P'
            kk_p = str(round(eval(eval_string), 4))
            ANOVA_P = str(round(eval(eval_stringANOVA), 4))

            if featurecopy in ordinals:
                trend = str(round(a, 5))

            if clock in immunes:
                if featurecopy in ordinals:
                    box.set_title('Cell values for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                else:
                    box.set_title('Cell values for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
            else:
                if clock == 'TelomerePC':
                    if featurecopy in ordinals:
                        box.set_title('Lengths for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                    else:
                        box.set_title('Lengths for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
                elif clock == 'DunedinPACE':
                    if featurecopy in ordinals:
                        box.set_title('Aging Rate for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p
                                      + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                    else:
                        box.set_title('Aging Rate for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
                else:
                    if featurecopy in ordinals:
                        box.set_title('Residuals for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p
                                      + ' || ANOVA p-Val: ' + ANOVA_P + ' || Trend: ' + trend)
                    else:
                        box.set_title('Residuals for ' + clock + ', ' + feat + '\nKruskal p-Val: ' + kk_p)
            if save_figs and num_samples > 15:
                if float(kk_p) < .05 or float(ANOVA_P) < .05:
                    plt.savefig('significant association figures/' + feat + '-' + clock + '.png')
                else:
                    plt.savefig('non-significant association figures/' + feat + '-' + clock + '.png')
            # if show_residuals:
            #     plt.show()

            plt.close()

            # def append_dfs():
            #     if list(noHRT_diffs) == list(HRT_diffs):
            #         pass
            #     else:
            #         if u == 'Menopausal no HRT':
            #             tt, ttp = scipy.stats.ttest_ind(HRT_diffs, noHRT_diffs, equal_var=False)
            #             try:
            #                 kk, kkp = stats.kruskal(HRT_diffs, noHRT_diffs)
            #             except:
            #                 kk, kkp = np.nan, np.nan
            #
            #             cat_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(HRT_diffs)
            #             cat_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(HRT_diffs)
            #             cat_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
            #             cat_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
            #             cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
            #             cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp
            #
            #     if u == 'Menopausal no HRT':
            #         binary_excluded_residuals = HRT_diffs
            #         unique_residuals = noHRT_diffs
            #     else:
            #         binary_excluded_residuals = noHRT_diffs
            #         unique_residuals = HRT_diffs
            #
            #     tt, ttp = scipy.stats.ttest_ind(unique_residuals, binary_excluded_residuals, equal_var=False)
            #     try:
            #         kk, kkp = stats.kruskal(unique_residuals, binary_excluded_residuals)
            #     except:
            #         kk, kkp = np.nan, np.nan
            #
            #     binary_exclude_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(unique_residuals)
            #     binary_exclude_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(unique_residuals)
            #     binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
            #     binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
            #     binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
            #     binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp

            # if feature == 'Menopause':
            #     is_male = covariates[covariates['Biological Sex'] == 'male'].index
            #     to_drop = (set(is_male) & set(vals.index))
            #     vals.drop(index=to_drop, inplace=True)
            #     is_HRT = covariates[['hormone replacement therapy' in str(x) for x in
            #                          covariates['Actively engage in anti-aging interventions?']]].index
            #     is_menopause = vals[vals == 'yes'].index
            #     is_menopause = (set(is_menopause) & set(differentials.index))
            #
            #     menoPositive_diffs = differentials.loc[is_menopause, clock]
            #
            #     is_HRT = (set(is_HRT) & set(menoPositive_diffs.index))
            #     HRT_diffs = menoPositive_diffs.loc[is_HRT]
            #     noHRT_diffs = menoPositive_diffs.drop(index=is_HRT)
            #
            #     kVal, k_P_Val = stats.kruskal(HRT_diffs, noHRT_diffs)
            #     ANVal, ANPVal = stats.f_oneway(HRT_diffs, noHRT_diffs)
            #     print(HRT_diffs, noHRT_diffs)
            #     fig = plt.figure()
            #     box = fig.add_subplot()
            #     box.boxplot([HRT_diffs, noHRT_diffs])
            #     box.set_xticklabels(['HRT', 'No HRT'])
            #
            #     avgs = [np.median(x) for x in [HRT_diffs, noHRT_diffs]]
            #     a, b = np.polyfit([1, 2], avgs, 1)
            #     box.plot(pd.Series([1, 2]), pd.Series([1, 2]) * a + b, 'r-')
            #
            #     box.set_title(
            #         'Residuals for ' + clock + ', ' + 'Women w Menopause, w + w/out HRT' + '\n Kruskal p-Val: ' + str(
            #             round(k_P_Val, 4)) + ' || ANOVA p-Val: ' + str(round(ANPVal, 4)) + ' || Trend: ' + str(
            #             round(a, 5)))
            #
            #     if show_residuals:
            #         plt.show()
            #     plt.savefig('Significant Association Figures/' + clock + '.png')
            #     plt.close()
            #
            #     for u in ['Menopausal no HRT', 'Menopausal with HRT']:
            #         if list(noHRT_diffs) == list(HRT_diffs):
            #             pass
            #         else:
            #             if u == 'Menopausal no HRT':
            #                 tt, ttp = scipy.stats.ttest_ind(HRT_diffs, noHRT_diffs, equal_var=False)
            #                 try:
            #                     kk, kkp = stats.kruskal(HRT_diffs, noHRT_diffs)
            #                 except:
            #                     kk, kkp = np.nan, np.nan
            #
            #                 cat_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(HRT_diffs)
            #                 cat_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(HRT_diffs)
            #                 cat_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
            #                 cat_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
            #                 cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
            #                 cat_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp
            #
            #         if u == 'Menopausal no HRT':
            #             binary_excluded_residuals = HRT_diffs
            #             unique_residuals = noHRT_diffs
            #         else:
            #             binary_excluded_residuals = noHRT_diffs
            #             unique_residuals = HRT_diffs
            #
            #         tt, ttp = scipy.stats.ttest_ind(unique_residuals, binary_excluded_residuals, equal_var=False)
            #         try:
            #             kk, kkp = stats.kruskal(unique_residuals, binary_excluded_residuals)
            #         except:
            #             kk, kkp = np.nan, np.nan
            #
            #         binary_exclude_associations.loc[feat + ': ' + u, 'Total Respondents'] = len(unique_residuals)
            #         binary_exclude_associations.loc[feat + ': ' + u, 'Avg ' + clock] = np.mean(unique_residuals)
            #         binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test'] = tt
            #         binary_exclude_associations.loc[feat + ': ' + u, clock + ' t-test p-Value'] = ttp
            #         binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis'] = kk
            #         binary_exclude_associations.loc[feat + ': ' + u, clock + ' Kruskal-Wallis p-Value'] = kkp
            # elif feature == 'Given Birth':
            #     is_male = covariates[covariates['Biological Sex'] == 'male'].index
            #     to_drop = (set(is_male) & set(vals.index))
            #     vals.drop(index=to_drop, inplace=True)

print('Final length of ass. columns: ', len(associations.columns))
print('Final length of cat-ass. columns: ', len(cat_associations.columns))

for c in associations.columns:
    if c not in og_cols:
        print(c)

if best_fit:
    associations.to_csv('Correlation+Relevance-BF.csv')
    cat_associations.to_csv('IndividualCorr+Rel-BF.csv')
    binary_exclude_associations.to_csv('BinaryExcludedCorr+Rel-BF.csv')
else:
    associations.to_csv('Correlation+Relevance.csv')
    cat_associations.to_csv('IndividualCorr+Rel.csv')
    binary_exclude_associations.to_csv('BinaryExcludedCorr+Rel.csv')

# associations = pd.read_csv('Correlation+Relevance.csv').set_index('Unnamed: 0')
# cat_associations = pd.read_csv('IndividualCorr+Rel.csv').set_index('Unnamed: 0')
# binary_exclude_associations = pd.read_csv('BinaryExcludedCorr+Rel.csv').set_index('Unnamed: 0')

significant_covariates = []
for cov in associations.index:
    significant = False
    for col in associations.columns:
        if significant:
            pass
        else:
            if 'p-Value' in str(col):
                try:
                    if associations.loc[cov, col] < .05:
                        significant_covariates.append(cov)
                        significant = True
                except:
                    pass
sig_associations = associations.loc[significant_covariates]
sig_associations.to_csv('SignificantCorr+Rel.csv')

significant_covariates = []
for cov in cat_associations.index:
    significant = False
    for col in cat_associations.columns:
        if significant:
            pass
        else:
            if 'p-Value' in str(col):
                if cat_associations.loc[cov, col] < .05:
                    significant_covariates.append(cov)
                    significant = True
sig_cat_associations = cat_associations.loc[significant_covariates]
sig_cat_associations.to_csv('SignificantIndividualCorr+Rel.csv')

significant_covariates = []
for cov in binary_exclude_associations.index:
    significant = False
    for col in binary_exclude_associations.columns:
        if significant:
            pass
        else:
            if 'p-Value' in str(col):
                if binary_exclude_associations.loc[cov, col] < .05:
                    significant_covariates.append(cov)
                    significant = True
sig_binary_exclude_associations = binary_exclude_associations.loc[significant_covariates]
sig_binary_exclude_associations.to_csv('SignificantBinaryExcludedCorr+Rel.csv')
