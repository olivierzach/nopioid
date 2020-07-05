import pandas as pd
import os
os.getcwd()
os.chdir('C:\\Users\\pklen\\Documents\\Sqlite\\data')
os.getcwd()
colmn = ['NPI', 'Last Name', 'First Name', 'Gender', 'Credential', 'Medical school name','Graduation year','Primary specialty','Secondary specialty 1','Secondary specialty 2','Organization legal name', 'Number of Group Practice members','Line 1 Street Address','Line 2 Street Address','City','State','Zip Code']
df = pd.read_csv("Physician_Compare_National_Downloadable_File.csv",usecols = colmn,low_memory=False)
print("End Read")

df.columns = ['NPI', 'Last_Name', 'First_Name', 'Gender', 'Credential', 'Medical_school_name','Graduation_year','Primary_specialty','Secondary_specialty_1','Secondary_specialty_2','Organization_legal_name', 'Number_of_Group_Practice_members','Line_1_Street_Address','Line_2_Street_Address','City','State','Zip_Code']

df_trimmed = df.sort_values('Number_of_Group_Practice_members', ascending=False).drop_duplicates('NPI').sort_index()
print("End-Sorting");

df_trimmed.to_csv("Physician.csv",encoding='utf-8',index=False)
print("End")
