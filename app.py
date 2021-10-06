import pandas as pd
import numpy as np
import base64
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import streamlit as st
import os
import zipfile
import shutil
from plot_multi_helper import plot_multi_helper
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)




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

    return df,dft,lst,df1,df2,df_Wellbore_development,df_Field_Reserves,df_Wellbore_Exploration_All


#Load data

df,dft,lst,df1Hist,df2Hist,df_Wellbore_development,df_Field_Reserves,df_Wellbore_Exploration_All = load_data()

#=================================================== Multiple oil ==================================
# Multiselect
#lstOil  = st.multiselect('Select fields for first production',lst,['EKOFISK','STATFJORD','TROLL'])

dfMultOil = df.copy()
# change prfInformationCarrier column name
dfMultOil.rename(columns={'prfInformationCarrier': 'Field'}, inplace=True)
dfMultOil_wells_filter = dfMultOil.copy()
#dfMultOil = dfMultOil[dfMultOil['Field'].isin(lstOil)].pivot(index='Years', columns='Field', values='prfPrdOilGrossMillSm3')

#=================================================== ============ ==================================

#================================================= Wellbore Exploration All rex code=========================
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
# dropdown selecttion
selection = st.selectbox('Select a field for detailed production analysis',lst)
userValue = selection

df_new = df[df['prfInformationCarrier'] == userValue]
dft_new = dft[dft['prfInformationCarrier'] == userValue]

df_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)
dft_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)


Columns = {'OIL':'prfPrdOilGrossMillSm3', 'GAS': 'prfPrdGasGrossBillSm3','CONDENSATE': 'prfPrdCondensateGrossMillSm3',
           'OE': 'prfPrdOeGrossMillSm3', 'WATER': 'prfPrdProducedWaterInFieldMillSm3' }

# dropdown Unite selection for Oil unit
uniteType_Oil = st.selectbox('Select oil production unit',['Sm3','STB'])

# dropdown Unite selection for Gas unit
uniteType_Gas = st.selectbox('Select gas production unit',['Sm3','ft3'])


columnNames = list(Columns.keys())


# Multiselect
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
# mergeng the two dataframes
dfHist = pd.merge(df1Hist, df2Hist, on='fldNpdidField')

# Extracting needed columns for charts
dfHist = dfHist[['fldName_x','fldInplaceOil', 'fldRecoverableOil', 'fldRemainingOil', 'fldRecoverableGas', 'fldRemainingGas', 'fldInplaceAssGas', 'fldInplaceFreeGas']]

# filtering with user value
df3Filtered = dfHist[dfHist['fldName_x'] == userValue]

# rename columns
df3Filtered = df3Filtered.rename(columns={'fldInplaceOil':'OIIP','fldRecoverableOil':'EUR_oil','fldRemainingOil':'NPD TRR_oil',
                   'fldRecoverableGas':'EUR_gas','fldRemainingGas':'NPD TRR_gas','fldInplaceAssGas':'GIIP_ass','fldInplaceFreeGas':'GIIP_free'})

# split the filterd dataframe into two dataframes one for oil and for gas
df3FilterdOil = df3Filtered[['OIIP','EUR_oil','NPD TRR_oil']]
df3FilterdGas = df3Filtered[['GIIP_free','GIIP_ass','EUR_gas','NPD TRR_gas']]


if uniteType_Oil == 'STB':
    df3FilterdOil = df3FilterdOil*6.2898

# convert the columns to rows for the bar chart (OIL)
df3FilterdOil_T = df3FilterdOil.T.reset_index()

# selecting the color palette (green)
color_base = sb.color_palette()[2]

with st.beta_expander('Display/hide histograms',True):
    col1,col2 = st.beta_columns(2)

    ax = sb.barplot(x = 'index',
                y = df3FilterdOil_T[df3FilterdOil_T.columns[1]],
                data = df3FilterdOil_T,
                color=color_base)
    ax.bar_label(ax.containers[0],fmt='%.2f');

    plt.title( userValue + ' Oil Volumes');
    plt.xlabel('');
    if uniteType_Oil == 'STB':
        plt.ylabel(' Oil Volume (MSTB)')
    else:
        plt.ylabel(' Oil Volume (MSm3)')

    # Show the plot
    plt.show()
    plt.xticks(fontsize=10)
    plt.savefig(final_directory + '/' + userValue + ' Oil Volumes.png')
    col1.pyplot()


    if uniteType_Gas == 'ft3':
        df3FilterdGas = df3FilterdGas*35.315

    # convert the columns to rows for the bar chart (GAS)
    df3FilterdGas_T = df3FilterdGas.T.reset_index()

    # selecting the color palette (blue)
    color_base = sb.color_palette()[3]

    ax = sb.barplot(x = 'index',
                y = df3FilterdGas_T[df3FilterdGas_T.columns[1]],
                data = df3FilterdGas_T,
                color = color_base)

    ax.bar_label(ax.containers[0],fmt='%.2f');

    plt.title(userValue + ' Gas Volumes');
    plt.xlabel('');
    if uniteType_Gas == 'ft3':
        plt.ylabel(' Gas Volume (Bft3)')
    else:
        plt.ylabel(' Gas Volume (BSm3)')

    # Show the plot
    plt.show()
    plt.xticks(fontsize=10)
    plt.savefig(final_directory + '/' + userValue + ' Gas Volumes.png')
    col2.pyplot()

#==========================================================================================================================================================================
#plot_multi2

groupORindiv = 'chose'
#plot_multi3

#plot_multi4

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
with st.beta_expander("Display/hide wells's status and content histogram",False):

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
    #=================================================================================================================================

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
with st.beta_expander("Display well's status histograms",False):
    from get_wellsCountDF import wells
    dfWellsAllFields,df_wells,wellsCountDF = wells().get_wellsCountDF(df_Wellbore_development,df_Field_Reserves)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    df_wells = df_wells[df_wells['fldName'] == userValue].set_index(['fldName'])

    st.text("Well's field information. The total number of wells:" + str(df_wells['wlbWellboreName'].nunique()))
    # Num of wells name that have Y in it
    pureY_df = df_wells[df_wells['wlbNamePart5'].notna()]
    df_wellsY = pureY_df[pureY_df['wlbNamePart5'].str.find('Y') !=-1]
    yCount = df_wellsY.shape[0]
    st.text("Number of wells planned as multilateral wellbores (Y):" + str(yCount))

    # dropdown status selecttion
    stlst = df_wells['wlbStatus'].unique()
    stselc = st.selectbox('Select a status to filtter with',stlst)
    st.dataframe(df_wells[df_wells['wlbStatus'] == stselc])
    # save the well count df with the main well
    #--------------------------------------------------------------
    from pandas import ExcelWriter
    w = ExcelWriter('Wells.xlsx')
    dfWellsAllFields.to_excel(w, sheet_name='Sheet0',index=False)
    wellsCountDF.to_excel(w, sheet_name='Sheet1',index=False)
    w.save()

    st.markdown(get_binary_file_downloader_html('Wells.xlsx', 'Well count and the main well data'), unsafe_allow_html=True)
    #--------------------------------------------------------------
    from plot_wells_status_purpose import plot_wells_status_purpose_content
    plot_wells_status_purpose_content().plot_contetnt(df_wells,userValue,final_directory)
    col1, col2 = st.beta_columns(2)
    plot_wells_status_purpose_content().plot_status(df_wells, userValue, final_directory,col1)
    plot_wells_status_purpose_content().plot_purpose(df_wells, userValue, final_directory,col2)

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

if (answer == 'individual' or answer =='both' or len(graphNum) ==1):
    lstdf = []
    for i in range(len(userValues)-1):
        dfcSum = df_new.copy()
        if ('OIL' in  userValues) & (uniteType_Oil =='STB'):
            dfcSum['OIL'] = dfcSum['OIL']*6.2898
        if ('GAS' in  userValues) & (uniteType_Gas =='ft3'):
            dfcSum['GAS'] = dfcSum['GAS']*35.315
        dfcSum = dfcSum[[userValues[-1],userValues[i]]]
        dfcSum.set_index('Years', inplace=True)
        dfcSum[userValues[i] + ' Cumulative'] = dfcSum[userValues[i]].cumsum()
        lstdf.append(dfcSum)

if st.button('Plot Individual Graphs'):
    st.header('Individual Graphs')
    groupORindiv = 'indiv'

    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and (len(userValues)-1>=1) :
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        if len(graphNum) ==1:
            userValuesclr.append('cumu')
            userValuesclr[1] = 'cumulative'
        else:
            userValuesclr[1] = 'cumulative'

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi3(groupORindiv,uniteType_Gas,uniteType_Oil,lstdf[0],userValuesclr,'yes', figsize=(20, 10));


        plt.title(str(userValue)+ ' Field ' + lstdf[0].columns.to_list()[0] + ' Production');

        plt.savefig(final_directory2 + '/' + userValue + ' Field ' + lstdf[0].columns.to_list()[0] + ' Production.png')
        st.pyplot()


    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=2:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[1]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)


        plot_multi_helper().plot_multi3(groupORindiv,uniteType_Gas,uniteType_Oil,lstdf[1],userValuesclr,'yes', figsize=(20, 10));

        plt.title(str(userValue)+ ' Field ' + lstdf[1].columns.to_list()[0] + ' Production');

        plt.savefig(final_directory2 + '/' + userValue + ' Field ' + lstdf[1].columns.to_list()[0] + ' Production.png')
        st.pyplot()

    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=3:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[2]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi3(groupORindiv,uniteType_Gas,uniteType_Oil,lstdf[2],userValuesclr,'yes', figsize=(20, 10));

        plt.title(str(userValue)+ ' Field ' + lstdf[2].columns.to_list()[0] + ' Production');

        plt.savefig(final_directory2 + '/' + userValue +  ' Field ' + lstdf[2].columns.to_list()[0] + ' Production.png')
        st.pyplot()


    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=4:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[3]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi3(groupORindiv,uniteType_Gas,uniteType_Oil,lstdf[3],userValuesclr,'yes', figsize=(20, 10));

        plt.title(str(userValue)+ ' Field ' + lstdf[3].columns.to_list()[0] + ' Production');

        plt.savefig(final_directory2 + '/' + userValue +  ' Field ' + lstdf[3].columns.to_list()[0] + ' Production.png')
        st.pyplot()


    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=5:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[4]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi3(groupORindiv,uniteType_Gas,uniteType_Oil,lstdf[4],userValuesclr,'yes', figsize=(20, 10));

        plt.title(str(userValue)+ ' Field ' + lstdf[4].columns.to_list()[0] + ' Production');

        plt.savefig(final_directory2 + '/' + userValue  + ' Field ' + lstdf[4].columns.to_list()[0] + ' Production.png')
        st.pyplot()


    #create plots download link
    zipf = zipfile.ZipFile('individual Plots.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('individual Plots', zipf)
    zipf.close()
    st.markdown(get_binary_file_downloader_html('individual Plots.zip', userValue + ' individual Plots'), unsafe_allow_html=True)
#===============================================================================================================================================================#
# Calculating GOR and CGR
dfCalc = df_new.copy()
if ('GAS' in userValues) & ('OIL' in userValues):
    dfCalc['GOR'] = ((dfCalc['GAS']*1000)/(dfCalc['OIL']))

if ('GAS' in userValues) & ('CONDENSATE' in userValues):
    dfCalc['CGR'] = ((dfCalc['CONDENSATE'])/(dfCalc['GAS']*1000))

if ('WATER' in userValues) & ('OIL' in userValues):
    dfCalc['WOR'] = ((dfCalc['WATER'])/(dfCalc['OIL']))
    dfCalc['WCUT'] = (((dfCalc['WATER'])/((dfCalc['OIL'])+(dfCalc['WATER']))))

dfCalc = dfCalc.fillna(0)
dfCalc.reset_index(drop=True, inplace=True)
dfCalc2 = dfCalc.copy()
dfCalc2.set_index('Years', inplace=True)


fluids = userValues.copy()

del fluids[-1]

dfCalc = dfCalc.drop(columns = fluids)


columnNamesCalc = dfCalc.columns.to_list()


lstdfCalc = []
for i in range(len(columnNamesCalc)-1):
    lstdfCalc.append(dfCalc[[columnNamesCalc[0],columnNamesCalc[i+1]]])

def calcIndex(lstdfCalc,name):
    for i in range(4):
        if lstdfCalc[i][lstdfCalc[i].columns[1]].name == name:
            return i

userVal = st.radio('Do you want to plot the GOR/CGR/WCUT with the fluids rates?',('Yes','No'))
userVal = userVal.lower()


#===============================================================================================================================================================#
#Ploting
if st.button('Plot Calculations Graphs'):
    st.header('Calculations Graphs')

    # Plot GOR
    if ('GAS' in userValues) & ('OIL' in userValues) & (userVal == 'no'):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'GOR')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Years').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)


        plt.title(str(userValue)+ ' Gas Oil Ratio');
        plt.xlabel('Years');
        plt.ylabel('Gas Oil Ratio (fraction)');
        ax.grid(axis='both', which='both')
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' Gas Oil Ratio.png')
        st.pyplot()

    elif ('GAS' in userValues) & ('OIL' in userValues) & (userVal == 'yes'):
        colorscalc = ['GOR','GAS','OIL']
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi4(dfCalc2[['GOR','GAS','OIL']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' Gas Oil Ratio');
        plt.xlabel('Years');
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' Gas Oil Ratio.png')
        st.pyplot()


    # Plot CGR
    if ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'no'):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'CGR')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Years').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)


        plt.title(str(userValue)+ ' CONDENSATE GAS Ratio');
        plt.xlabel('Years');
        plt.ylabel('CONDENSATE GAS Ratio (fraction)');
        ax.grid(axis='both', which='both')
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' CONDENSATE GAS Ratio.png')
        st.pyplot()

    elif ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'yes'):
        colorscalc = ['CGR','GAS','CONDENSATE']
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi4(dfCalc2[['CGR','GAS','CONDENSATE']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' CONDENSATE GAS Ratio');
        plt.xlabel('Years');
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' CONDENSATE GAS Ratio.png')
        st.pyplot()


    # Polt WOR
    if ('WATER' in userValues) & ('OIL' in userValues):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'WOR')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Years').plot(figsize=(20,10), color ='purple',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)


        plt.title(str(userValue)+ ' Water Oil Ratio');
        plt.xlabel('Years');
        plt.ylabel('Water Oil Ratio (fraction)');
        ax.grid(axis='both', which='both')
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' Water Oil Ratio.png')
        st.pyplot()


    # Polt WCUT
    if ('WATER' in userValues) & ('OIL' in userValues) & (userVal == 'no'):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'WCUT')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Years').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)


        plt.title(str(userValue)+ ' Water Cut');
        plt.xlabel('Years');
        plt.ylabel('Water Cut (fraction)');
        ax.grid(axis='both', which='both')
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' Water Cut.png')
        st.pyplot()

    elif ('GAS' in userValues) & ('WATER' in userValues) & (userVal == 'yes'):
        colorscalc = ['WCUT','OIL','WATER',]
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi_helper().plot_multi4(dfCalc2[['WCUT','OIL','WATER']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' Water Cut');
        plt.xlabel('Years');
        plt.savefig(final_directory3 + '/' + str(userValue)+ ' Water Cut.png')
        st.pyplot()

    #create plots download link
    zipf = zipfile.ZipFile('Calculation Plots.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('Calculation Plots', zipf)
    zipf.close()

    st.markdown(get_binary_file_downloader_html('Calculation Plots.zip', userValue + ' Calculation Plots'), unsafe_allow_html=True)

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

st.markdown(DownloadFunc(df_all), unsafe_allow_html=True)
