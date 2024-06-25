import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

covariates = pd.read_csv('CovariatesCleaned.csv').set_index('PatientID')

continuous = ['Weight',
              'Height',
              'Age for Loss of Hair',
              'Stress Level',
              'Pack Years (If smoker)']

# heights = covariates['Height']
# curated_values = []
# removed_values = []
# for height in heights:
#     bv = False
#     if 'in in' in height:
#         covariates.replace(height, height.replace('in in', 'in'), inplace=True)
#         curated_values.append(height)
#         height = height.replace('in in', 'in')
#     elif 'i in'in height:
#         covariates.replace(height, height.replace('i in', 'in'), inplace=True)
#         curated_values.append(height)
#         height = height.replace('i in', 'in')
#
#     feet, inches = height.split(' ')[0], height.split(' ')[-2]
#     try:
#         float(feet), float(inches)
#     except:
#         try:
#             inches = inches.split(',')[-1]
#             float(feet), float(inches)
#         except Exception as e:
#             print('Bad val: ', e, height)
#             print(feet, inches)
#             covariates.replace(height, 'remove this value', inplace=True)
#             curated_values.append(height)
#             bv = True
#
#     if not bv:
#         if inches in ['i', 'in', 'ft']:
#             inches = height.split(' ')[-3].split(',')[-1]
#             curated_values.append(height)
#         elif "'" in str(inches):
#             covariates.replace(height, 'remove this value', inplace=True)
#             removed_values.append(height)
#         elif feet == inches:
#             try:
#                 new_height = float(feet) * 12 + float(inches)
#                 covariates.replace(height, new_height, inplace=True)
#                 curated_values.append(height)
#             except:
#                 covariates.replace(height, 'remove this value', inplace=True)
#                 removed_values.append(height)
#         elif inches == '' or feet == '':
#             covariates.replace(height, 'remove this value', inplace=True)
#             removed_values.append(height)
#         elif float(feet) > 7 or float(inches) > 12:
#             try:
#                 if int(str(feet)[0]) * 12 + int(str(feet)[1]) == inches:
#                     feet, inches = int(str(feet)[0]), int(str(feet)[1])
#                     curated_values.append(height)
#                 else:
#                     covariates.replace(height, 'remove this value', inplace=True)
#                     removed_values.append(height)
#             except:
#                 if float(inches) - 12 < float(feet) * 12 < float(inches) + 12:
#                     covariates.replace(height, inches, inplace=True)
#                     curated_values.append(height)
#                 else:
#                     covariates.replace(height, 'remove this value', inplace=True)
#                     removed_values.append(height)
#         else:
#             new_height = float(feet) * 12 + float(inches)
#             # print(new_height)
#             covariates.replace(height, new_height, inplace=True)
#
# post_heights = covariates['Height'].replace('remove this value').astype(float)
# for value in post_heights:
#     if value < 48 or value > 88:
#         covariates.replace(value, 'remove this value', inplace=True)
# post_heights = covariates['Height'].replace('remove this value').astype(float)
#
# print('Number removed: ', list(covariates['Height']).count('remove this value'))
# print(np.mean(post_heights),
#       np.min(post_heights),
#       np.max(post_heights))
# print('Number of curated values and all curated values: ', len(curated_values))
# print(curated_values)
# print('Number of removed values and all removed values: ', len(removed_values))
# print(removed_values)
# covariates.to_csv('CovariatesCleaned.csv')

# for py in covariates['Pack Years (If smoker)'].fillna(0).values:
#     try:
#         if float(py) > 100:
#             covariates['Pack Years (If smoker)'].replace(py, 'remove this value', inplace=True)
#     except Exception as e:
#         print(e)
# for weight in covariates['Weight'].fillna(0).values:
#     try:
#         if float(weight) > 500 or float(weight) < 40:
#             covariates['Weight'].replace(weight, 'remove this value', inplace=True)
#     except Exception as e:
#         print(e)
#
#
# for covariate in continuous:
#     print(covariate)
#     values = np.array(covariates[covariate])
#     if covariate == 'Weight':
#         for val in values:
#             if ',' in str(val):
#                 covariates.replace(val, str(val).replace(',', '.'), inplace=True)
#     values = list(covariates[covariate])
#     print(values)
#     try:
#         for v in range(values.count('remove this value')):
#             values.remove('remove this value')
#     except Exception as e:
#         print(e)
#
#     print(values)
#     values = np.array(values).astype(float)
#     values = values[np.logical_not(np.isnan(values))]
#     if covariate == 'Age for Loss of Hair':
#         values = values[np.logical_not(values == 0)]
#
#     print(values)
#
#     avg, med, minimum, maximum = np.nanmean(values), np.nanmedian(values), np.nanmin(values), np.nanmax(values)
#     print(avg, med, med, minimum, maximum)
#
#     # fig = plt.figure()
#     # box = fig.add_subplot(111)
#     # box.boxplot(values)
#     # box.set_title(covariate)
#     # plt.show()
#
# covariates.to_csv('CovariatesCleaned NEW.csv')

cleaned_list = ['NeuropsychologicalDiagnosis_CovariateData.csv',
                'NutritionalSupplements_CovariateData.csv',
                'RecDrugUse_CovariateData.csv',
                # 'ReproductiveDiagnosis_CovariateData.csv',
                'RespiratoryDiagnosis_Covariate Data.csv',
                'SpecialMed_CovariateData.csv',
                'Medicine_CovariateData',
                'ImmuneDiagnosis_CovarariateData',
                'GenitoUrinaryDiagnosis_CovariateData',
                'GastroDiagnosis_CovariateData',
                'FamHis_CovariateData',
                'Ethnicity_CovariateData',
                'EndocrineDiagnosis_CovariateData',
                'Diet_CovariateData',
                'CardioDiagnosis_CovariateData',
                'BMI_CovariateData']

# exit()


for cov in os.listdir('Separate Covariates'):
    var_name = cov.split('_')[0]
    exec(var_name + ' = pd.read_excel("Separate Covariates/' + cov + '")')

    if var_name == 'BMI':
        exec(var_name + ' = ' + var_name + '.set_index("PatientID")["BMI"]')
    elif var_name in ['CardioDiagnosis',
                      'Diet',
                      'EndocrineDiagnosis',
                      'Ethnicity',
                      'MusculoskeletalDiagnosis',
                      'NutritionalSupplements',
                      'ReproductiveDiagnosis']:
        exec(var_name + '.set_index(' + var_name + '.columns[0], inplace=True)')
        if var_name == 'EndocrineDiagnosis':
            eval(var_name + '.drop(columns=' + var_name + '.columns[3], inplace=True)')
    else:
        exec(var_name + ' = ' + var_name + '.transpose()')
        exec(var_name + '.set_index(' + var_name + '.columns[1], inplace=True)')


    eval('print("new table", var_name)')
    exec('if isinstance(' + var_name + ', pd.Series):' +
                                       'print(eval(var_name))' +
                                       '\nelse:' +
                                       '[print(eval(var_name))]')  # [col]) for col in eval(var_name).columns[:2]]')


