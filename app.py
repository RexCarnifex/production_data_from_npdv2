import pandas as pd
import numpy as np
import base64
import streamlit as st
import os
import shutil
import time
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


# create file for csvs
current_directory = os.getcwd()
csvs_directory = os.path.join(current_directory, r'csvs')
if not os.path.exists(csvs_directory):
   os.makedirs(csvs_directory)


@st.cache
def load_data():
    df = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_production_gross_monthly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    df1 = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_reserves&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    df2 = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_in_place_volumes&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    # Discovery Overview
    df_Wellbore_development = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/wellbore_development_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')
    #
    # Field - Reserves
    df_Field_Reserves = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_reserves&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')
    
    df_Wellbore_Exploration_All = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/wellbore_exploration_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    #df = pd.read_csv('data.csv')
    # make a new time column
    df['Years'] = df['prfYear'].astype(str) + '-' + df['prfMonth'].astype(str)

    # covert the object datatype to date
    df['Years'] = pd.to_datetime(df['Years'])
    dft = df.copy()

    df['Years'] = df['Years'].dt.to_period('M')

    # remove white spaces at the end of each entry in the prfInformationCarrier column (if any)
    df['prfInformationCarrier'] = df['prfInformationCarrier'].str.lstrip()
    dft['prfInformationCarrier'] = dft['prfInformationCarrier'].str.lstrip()

    # remove white spaces at the begining of each entry in the prfInformationCarrier column (if any)
    df['prfInformationCarrier'] = df['prfInformationCarrier'].str.rstrip()
    dft['prfInformationCarrier'] = dft['prfInformationCarrier'].str.rstrip()


    # constraction a list for the values in the prfInformationCarrier column
    lst = df['prfInformationCarrier'].value_counts().index.unique().to_list()
    i = lst.index("TROLL")
    lst[i], lst[0] = lst[0], lst[i]

    df.to_csv(csvs_directory + '/' + 'df.csv', index=None, header=True)
    dft.to_csv(csvs_directory + '/' + 'dft.csv', index=None, header=True)
    df1.to_csv(csvs_directory + '/' + 'df1Hist.csv', index=None, header=True)
    df2.to_csv(csvs_directory + '/' + 'df2Hist.csv', index=None, header=True)
    df_Wellbore_development.to_csv(csvs_directory + '/' + 'df_Wellbore_development.csv', index=None, header=True)
    df_Field_Reserves.to_csv(csvs_directory + '/' + 'df_Field_Reserves.csv', index=None, header=True)
    df_Wellbore_Exploration_All.to_csv(csvs_directory + '/' + 'df_Wellbore_Exploration_All.csv', index=None, header=True)

    return df,dft,df1,df2,df_Wellbore_development,df_Field_Reserves,df_Wellbore_Exploration_All

#st.text(os.listdir(csvs_directory))
#Load data
@st.cache
def read_data():
    if len(os.listdir(csvs_directory)) == 0:
        df,dft,df1Hist,df2Hist,df_Wellbore_development,df_Field_Reserves,df_Wellbore_Exploration_All = load_data()
    else:

        df = pd.read_csv(csvs_directory + '/' + 'df.csv')
        dft = pd.read_csv(csvs_directory + '/' + 'dft.csv')
        df1Hist = pd.read_csv(csvs_directory + '/' + 'df1Hist.csv')
        df2Hist = pd.read_csv(csvs_directory + '/' + 'df2Hist.csv')
        df_Wellbore_development = pd.read_csv(csvs_directory + '/' + 'df_Wellbore_development.csv')
        df_Field_Reserves = pd.read_csv(csvs_directory + '/' + 'df_Field_Reserves.csv')
        df_Wellbore_Exploration_All = pd.read_csv(csvs_directory + '/' + 'df_Wellbore_Exploration_All.csv')

    return df, dft, df1Hist, df2Hist, df_Wellbore_development, df_Field_Reserves, df_Wellbore_Exploration_All


#if nav_selection == 'Welcome':
df,dft,df1Hist,df2Hist,df_Wellbore_development,df_Field_Reserves,df_Wellbore_Exploration_All = read_data()

lst = df['prfInformationCarrier'].value_counts().index.unique().to_list()
#=================================================== Multiple oil ==================================
# Multiselect
#lstOil  = st.multiselect('Select fields for first production',lst,['EKOFISK','STATFJORD','TROLL'])

dfMultOil = df.copy()
# change prfInformationCarrier column name
dfMultOil.rename(columns={'prfInformationCarrier': 'Field'}, inplace=True)
dfMultOil_wells_filter = dfMultOil.copy()
#dfMultOil = dfMultOil[dfMultOil['Field'].isin(lstOil)].pivot(index='Years', columns='Field', values='prfPrdOilGrossMillSm3')

#=================================================== ============ ==================================

#================================================= Wellbore Exploration All =========================
# drop empty values in fldNpdidField column
df_Wellbore_Exploration_All = df_Wellbore_Exploration_All[df_Wellbore_Exploration_All['fldNpdidField'].notna()]

# change fldNpdidField type to int so it matches the other two dataframes
df_Wellbore_Exploration_All['fldNpdidField'] = df_Wellbore_Exploration_All['fldNpdidField'].astype(int)

# Selecting columns to be used in analysis
df_new = df_Wellbore_Exploration_All[['wlbWellboreName','wlbDrillingOperator', 'fldNpdidField', 'wlbProductionLicence', 'wlbStatus', 'wlbWellType', 'wlbContent', 'wlbMainArea', 'wlbFormationWithHc1','wlbAgeWithHc1','wlbFormationWithHc2','wlbAgeWithHc2','wlbFormationWithHc3','wlbAgeWithHc3']]

# Merge df_new on df_Field_Reserves (Field Reserves to get FieldID and Field Names)
df_Wellbore_Exploration_All_and_Reserves = pd.merge(df_new, df_Field_Reserves, how='left', on='fldNpdidField')

# Removing All conlumns except for fldname
df_Wellbore_Exploration_All_and_Reserves.drop(['fldRecoverableOil','fldRecoverableGas','fldRecoverableNGL','fldRecoverableCondensate','fldRecoverableOE','fldRemainingOil','fldRemainingGas','fldRemainingNGL','fldRemainingCondensate','fldRemainingOE','fldDateOffResEstDisplay','DatesyncNPD'], axis=1, inplace=True)
#===================================================================================================
nav_selection = st.sidebar.radio('Navigation',['Welcome','Individual Analysis','Group Analysis'])


if nav_selection == 'Welcome':
    st.title('Welcome to oil analysis app')
    st.text('this app will help in..............')


userValue = 'TROLL'
# dropdown selecttion
if nav_selection == 'Individual Analysis' or nav_selection == 'Group Analysis':
    selection = st.selectbox('Select a field for detailed production analysis',lst)
    userValue = selection

df_new = df[df['prfInformationCarrier'] == userValue]
dft_new = dft[dft['prfInformationCarrier'] == userValue]


df_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)
dft_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)


Columns = {'OIL':'prfPrdOilGrossMillSm3', 'GAS': 'prfPrdGasGrossBillSm3','CONDENSATE': 'prfPrdCondensateGrossMillSm3',
           'OE': 'prfPrdOeGrossMillSm3', 'WATER': 'prfPrdProducedWaterInFieldMillSm3' }

uniteType_Oil = 'Sm3'
uniteType_Gas = 'Sm3'
if nav_selection == 'Individual Analysis' or nav_selection == 'Group Analysis':
    # dropdown Unite selection for Oil unit
    uniteType_Oil = st.selectbox('Select oil production unit',['Sm3','STB'])

    # dropdown Unite selection for Gas unit
    uniteType_Gas = st.selectbox('Select gas production unit',['Sm3','ft3'])


columnNames = list(Columns.keys())


# Multiselect
graphNum = ['OIL', 'GAS', 'CONDENSATE', 'OE', 'WATER']
if nav_selection == 'Group Analysis':
    graphNum  = st.multiselect('Select The Fluids to Plot',['OIL', 'GAS', 'CONDENSATE', 'OE', 'WATER'],['OIL', 'GAS', 'CONDENSATE', 'OE', 'WATER'])

userValues = graphNum
userValues = list(map(str.upper,userValues))


userValues.append('Years')


# extract only wanted data from ploting
df_new = df_new[userValues]
dft_new = dft_new[userValues]
df_new.reset_index(drop=True, inplace=True)


dft_new.set_index('Years', inplace=True)

# create file for images
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'Group Plots')
final_directory2 = os.path.join(current_directory, r'Individual Plots')
final_directory3 = os.path.join(current_directory, r'Calculation Plots')

if os.path.exists(final_directory):
    shutil.rmtree(final_directory)
if os.path.exists(final_directory2):
    shutil.rmtree(final_directory2)
if os.path.exists(final_directory3):
    shutil.rmtree(final_directory3)

if not os.path.exists(final_directory):
   os.makedirs(final_directory)
if not os.path.exists(final_directory2):
   os.makedirs(final_directory2)
if not os.path.exists(final_directory3):
   os.makedirs(final_directory3)

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

# download plots
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

#============================================== Hist ============================================================================================================================
if nav_selection == 'Individual Analysis':
    from individual import Indivdual
    Indivdual().histogram(df1Hist,df2Hist,userValue,uniteType_Oil,final_directory,uniteType_Gas)

#==========================================================================================================================================================================
#plot_multi2

groupORindiv = 'chose'
#plot_multi3

#plot_multi4
ans = 'No'
if nav_selection == 'Individual Analysis' or nav_selection == 'Group Analysis':
    ans = st.radio('Select time interval for plotting?',('No','Yes'))

if ans.lower() == 'yes':
    statrtTime = st.text_input("Enter a Start year: ", '1990')
    endTime = st.text_input("Enter an End year: ", '2022')

    statrtTime = int(statrtTime) -1
    statrtTime = str(statrtTime) + '-' + '12'
    endTime = endTime + '-' + '02'
    df_new = df_new[(df_new['Years']> statrtTime) & (df_new['Years']< endTime)]
    dft_new = dft_new[(dft_new.index> statrtTime) & (dft_new.index< endTime)]
    dfMultOil = dfMultOil[(dfMultOil.index> statrtTime) & (dfMultOil.index< endTime)]
    dfMultOil_wells_filter = dfMultOil_wells_filter[(dfMultOil_wells_filter.index > statrtTime) & (dfMultOil_wells_filter.index < endTime)]
    df_new.reset_index(drop=True, inplace=True)


answer = 'both'
yearsx = df_new['Years'].dt.year.to_list()
yearsx = list(set(yearsx))

df_newcSUM = df_new.copy()
for i in range(len(userValues)-1):
    df_newcSUM[userValues[i] + ' Cumulative Production'] = df_newcSUM[userValues[i]].cumsum()

dftt_newcSUM = dft_new.copy()
for i in range(len(userValues)-1):
    dftt_newcSUM[userValues[i] + ' Cumulative Production'] = dftt_newcSUM[userValues[i]].cumsum()

# show table
if nav_selection == 'Individual Analysis':
    Numrows = st.text_input("Display the last months of production data.", '5')
    st.text('Last ' + Numrows + ' rows of Filtered Data')

    if uniteType_Oil == 'STB' and uniteType_Gas == 'ft3':
        dftt_newcSUMoilGas = dftt_newcSUM.copy()
        dftt_newcSUMoilGas[['OIL','OIL Cumulative Production']] = dftt_newcSUMoilGas[['OIL','OIL Cumulative Production']]*6.2898
        dftt_newcSUMoilGas[['GAS','GAS Cumulative Production']] = dftt_newcSUMoilGas[['GAS','GAS Cumulative Production']]*35.315
        st.dataframe(dftt_newcSUMoilGas.tail(int(Numrows)))
    elif uniteType_Oil == 'STB' and not uniteType_Gas == 'ft3':
        dftt_newcSUMoil = dftt_newcSUM.copy()
        dftt_newcSUMoil[['OIL','OIL Cumulative Production']] = dftt_newcSUMoil[['OIL','OIL Cumulative Production']]*6.2898
        st.dataframe(dftt_newcSUMoil.tail(int(Numrows)))
    elif uniteType_Gas == 'ft3' and not uniteType_Oil == 'STB':
        dftt_newcSUMgas = dftt_newcSUM.copy()
        dftt_newcSUMgas[['GAS','GAS Cumulative Production']] = dftt_newcSUMgas[['GAS','GAS Cumulative Production']]*35.315
        st.dataframe(dftt_newcSUMgas.tail(int(Numrows)))
    else:
        st.dataframe(dftt_newcSUM.tail(int(Numrows)))


#--------------------------------------------------------------------------------------------------------------
# description part
if nav_selection == 'Individual Analysis':
    dfINFO = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_description&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')
    dfINFO = dfINFO[['fldName','fldDescriptionHeading' ,'fldDescriptionText']]

    dfINFO1 = dfINFO.pivot(index='fldName', columns='fldDescriptionHeading', values='fldDescriptionText')
    dfINFO1.reset_index(inplace=True)

    d = dfINFO1[dfINFO1['fldName'] == userValue]
    d.drop(columns = ['Recovery strategy '], inplace=True)
    d.set_index(['fldName'],drop=True,inplace=True)

    # show table
    #st.text('Description Data')
    #st.dataframe(d)

    # creating 5 columns of text to show description
    with st.beta_expander('Display/hide NPD field description',False):
        col1,col2,col3,col4,col5 = st.beta_columns(5)
        col1.markdown("<h1 style='text-align: center; font-size:20px;'>Development</h1>", unsafe_allow_html=True)
        col1.success(str(d['Development '].values[0]))

        col2.markdown("<h1 style='text-align: center; font-size:20px;'>Recovery</h1>", unsafe_allow_html=True)
        col2.success(str(d['Recovery '].values[0]))

        col3.markdown("<h1 style='text-align: center; font-size:20px;'>Reservoir</h1>", unsafe_allow_html=True)
        col3.success(str(d['Reservoir '].values[0]))

        col4.markdown("<h1 style='text-align: center; font-size:20px;'>Status</h1>", unsafe_allow_html=True)
        col4.success(str(d['Status '].values[0]))

        col5.markdown("<h1 style='text-align: center; font-size:20px;'>Transport</h1>", unsafe_allow_html=True)
        col5.success(str(d['Transport '].values[0]))

#--------------------------------------------------------------------------------------------------------------
# Show wells table
#========================================================================================================================================
#with st.beta_expander("Display/hide wells's status and content histogram",False):
if nav_selection == 'Group Analysis':
    wantedlst = ['fldName', 'wlbMainArea', 'wlbFormationWithHc1' ,   'wlbAgeWithHc1'   , 'wlbFormationWithHc2'   , 'wlbAgeWithHc2'  ,  'wlbFormationWithHc3'  ,  'wlbAgeWithHc3']
    df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves[wantedlst]
    df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves.set_index('fldName')
    wlbMainAreaValues = list(df_Wellbore_Exploration_All_and_Reserves['wlbMainArea'].unique())

    # dropdown selecttion
    #fieldslst = list(df_Wellbore_Exploration_All_and_Reserves.index.unique())
    #selectedfield = st.selectbox('Select a Field to filtter with',fieldslst)
    #df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves[df_Wellbore_Exploration_All_and_Reserves.index == selectedfield]

    st.text('Check wanted NCS segments')
    option1 = st.checkbox(wlbMainAreaValues[0])
    option2 = st.checkbox(wlbMainAreaValues[1])
    option3 = st.checkbox(wlbMainAreaValues[2])

    # remove white spaces at the end and begining of each entry (if any)
    for column in wantedlst[2:]:
        df_Wellbore_Exploration_All_and_Reserves[column] = df_Wellbore_Exploration_All_and_Reserves[column].str.lstrip()
        df_Wellbore_Exploration_All_and_Reserves[column] = df_Wellbore_Exploration_All_and_Reserves[column].str.rstrip()

    df_Wellbore_Exploration_All_and_Reserves.replace('',np.nan,inplace = True)

    col1,col2,col3 = st.beta_columns(3)

    formations1 = list(df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc1'].dropna().unique())
    formations1Selected  = col1.multiselect('Select wanted formations 1',formations1)
    if len(formations1Selected) >0:
        df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves[df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc1'].isin(formations1Selected)]

    formations2 = list(df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc2'].dropna().unique())
    formations2Selected  = col2.multiselect('Select wanted formations 2',formations2)
    if len(formations2Selected) >0:
        df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves[df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc2'].isin(formations2Selected)]

    formations3 = list(df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc3'].dropna().unique())
    formations3Selected  = col3.multiselect('Select wanted formations 3',formations3)
    if len(formations3Selected) >0:
        df_Wellbore_Exploration_All_and_Reserves = df_Wellbore_Exploration_All_and_Reserves[df_Wellbore_Exploration_All_and_Reserves['wlbFormationWithHc3'].isin(formations3Selected)]


    #filterdWells = df_Wellbore_Exploration_All_and_Reserves[df_Wellbore_Exploration_All_and_Reserves.index == userValue]
    filterdWells = df_Wellbore_Exploration_All_and_Reserves.copy()
    if option1 and not option2 and not option3:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[0]])]
    if option2 and not option1 and not option3:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[1]])]
    if option3 and not option2 and not option1:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[2]])]

    if option1 and option2 and not option3:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[0],wlbMainAreaValues[1]])]
    if option1 and option3 and not option2:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[0],wlbMainAreaValues[2]])]
    if option1 and option2 and option3:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[0],wlbMainAreaValues[1],wlbMainAreaValues[2]])]

    if option2 and option3 and not option1:
        filterdWells = filterdWells[filterdWells['wlbMainArea'].isin([wlbMainAreaValues[1],wlbMainAreaValues[2]])]

    filterdWells.replace(np.nan,'',inplace = True)
    if len(formations1Selected) >0 or len(formations2Selected) >0 or len(formations3Selected) >0:
        st.dataframe(filterdWells)
        from get_wellsCountDF import wells
        choosen_filtered_fields = wells().plot_multi_oil(filterdWells.reset_index(),dfMultOil_wells_filter,uniteType_Oil,final_directory)
    else:
        choosen_filtered_fields = []
    #=================================================================================================================================

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if nav_selection == 'Individual Analysis':
    from individual import Indivdual
    Indivdual().wells_status_histograms(df_Wellbore_development,df_Field_Reserves,userValue,final_directory)

#=======================================================================================================================================================
dfcum = df_newcSUM.copy()

userValuescSum = userValues.copy()
del userValuescSum[-1]

df_newcSUM = df_newcSUM.drop(columns = userValuescSum)
dftt_newcSUM = dftt_newcSUM.drop(columns = userValuescSum)



csumNames = df_newcSUM.columns.to_list()
if 'GAS Cumulative Production' in csumNames:
    del csumNames[csumNames.index('GAS Cumulative Production')]


mfluids = userValues.copy()
if 'GAS' in mfluids:
    del mfluids[mfluids.index('GAS')]
mcolors = mfluids.copy()
del mcolors[-1]

for fluid in mfluids:
        if fluid == 'OIL':
            mcolors[mfluids.index('OIL')] = 'green'
        elif fluid == 'WATER':
            mcolors[mfluids.index('WATER')] = 'blue'
        elif fluid == 'OE':
            mcolors[mfluids.index('OE')] = 'black'
        elif fluid == 'CONDENSATE':
            mcolors[mfluids.index('CONDENSATE')] = 'orange'
#===============================================================================================================================================================#
if nav_selection == 'Group Analysis':
    if st.button('Plot Group Graphs - Years'):
        st.header('Group Graphs')
        groupORindiv = 'group'
        from group_plot import group_plot
        group_plot().plot(dfMultOil,choosen_filtered_fields,'years',userValues,userValue,uniteType_Oil,graphNum,final_directory,df_new,dft_new,answer,csumNames,mcolors,df_newcSUM,dftt_newcSUM,yearsx,mfluids,groupORindiv,uniteType_Gas)
        #===============================================================================================================================================================#

    if st.button('Plot Group Graphs - Months'):
        st.header('Group Graphs')
        groupORindiv = 'group'
        from group_plot import group_plot
        group_plot().plot(dfMultOil,choosen_filtered_fields,'months',userValues,userValue,uniteType_Oil,graphNum,final_directory,df_new,dft_new,answer,csumNames,mcolors,df_newcSUM,dftt_newcSUM,yearsx,mfluids,groupORindiv,uniteType_Gas)
        #===============================================================================================================================================================#

if nav_selection == 'Individual Analysis':
    from individual import Indivdual
    lstdf = Indivdual().comulative_calcuations(answer,df_new,uniteType_Oil,userValues,uniteType_Gas,graphNum)

    # Indivdual plot
    Indivdual().plot_indivdual(answer,graphNum,userValues,yearsx,uniteType_Gas,uniteType_Oil,lstdf,userValue,final_directory2)
#===============================================================================================================================================================#
    # Calculating GOR and CGR and plotting
    Indivdual().calculations_GOR_CGR_plot(df_new,userValues,yearsx,userValue,final_directory3)
from individual import Indivdual
dfCalc,dfCalc2 = Indivdual().get_dfCalc(userValues,df_new)
#==============================================================================================================================================================
# Export data
if 'GAS' in userValues:
    df_new.set_index(['Years','GAS'],inplace=True)
    df_new.columns += ' [MSm3]'
    df_new.reset_index(inplace=True)
    df_new.rename(columns = {'GAS':'GAS [BSm3]'},inplace=True)
else:
    df_new.set_index(['Years'],inplace=True)
    df_new.columns += ' [MSm3]'
    df_new.reset_index(inplace=True)

df_all = pd.merge(df_new,df_newcSUM, on='Years')
df_all = pd.merge(df_all,dfCalc, on='Years')

def DownloadFunc(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{userValue}.csv">Download the data from ' + userValue + ' field as csv file</a>'
    return href

if nav_selection == 'Individual Analysis' or nav_selection == 'Group Analysis':
    st.markdown(DownloadFunc(df_all), unsafe_allow_html=True)
