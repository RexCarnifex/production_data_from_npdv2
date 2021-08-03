import pandas as pd
import numpy as np
import base64
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache
def load_data():
    df = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_production_gross_monthly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    df1 = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_reserves&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    df2 = pd.read_csv('https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_in_place_volumes&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false')

    #df = pd.read_csv('data.csv')
    # make a new time column
    df['Time'] = df['prfYear'].astype(str) + '-' + df['prfMonth'].astype(str)

    # covert the object datatype to date
    df['Time'] = pd.to_datetime(df['Time'])
    dft = df.copy()

    df['Time'] = df['Time'].dt.to_period('M')

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

    return df,dft,lst,df1,df2


#Load data
df,dft,lst,df1Hist,df2Hist = load_data()

#=================================================== Multiple oil ==================================
# Multiselect
lstOil  = st.multiselect('For Multiple oil graph select the fields to plot with',lst,['EKOFISK','STATFJORD','TROLL'])

dfMultOil = df.copy()
# change prfInformationCarrier column name
dfMultOil.rename(columns={'prfInformationCarrier': 'Field'}, inplace=True)
dfMultOil = dfMultOil[dfMultOil['Field'].isin(lstOil)].pivot(index='Time', columns='Field', values='prfPrdOilGrossMillSm3')

#=================================================== ============ ==================================


# taking input from the user and checking if it's in prfInformationCarrier column
# dropdown selecttion
selection = st.selectbox('Select a Field to filtter with',lst) 
userValue = selection

df_new = df[df['prfInformationCarrier'] == userValue]
dft_new = dft[dft['prfInformationCarrier'] == userValue]

df_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)
dft_new.rename(columns={'prfPrdOilGrossMillSm3': 'OIL', 'prfPrdGasGrossBillSm3':'GAS','prfPrdCondensateGrossMillSm3':'CONDENSATE','prfPrdOeGrossMillSm3':'OE','prfPrdProducedWaterInFieldMillSm3':'WATER'  }, inplace=True)


Columns = {'OIL':'prfPrdOilGrossMillSm3', 'GAS': 'prfPrdGasGrossBillSm3','CONDENSATE': 'prfPrdCondensateGrossMillSm3',
           'OE': 'prfPrdOeGrossMillSm3', 'WATER': 'prfPrdProducedWaterInFieldMillSm3' }


columnNames = list(Columns.keys())


# Multiselect
graphNum  = st.multiselect('Select The Fluids to Plot',['OIL', 'GAS', 'CONDENSATE', 'OE', 'WATER'],['OIL', 'GAS', 'CONDENSATE', 'OE', 'WATER'])

userValues = graphNum
userValues = list(map(str.upper,userValues))


userValues.append('Time')


# extract only wanted data from ploting
df_new = df_new[userValues]
dft_new = dft_new[userValues]
df_new.reset_index(drop=True, inplace=True)


dft_new.set_index('Time', inplace=True)

#============================================== Hist ========================================
# mergeng the two dataframes
dfHist = pd.merge(df1Hist, df2Hist, on='fldNpdidField')

# Extracting needed columns for charts
dfHist = dfHist[['fldName_x','fldInplaceOil', 'fldRecoverableOil', 'fldRemainingOil', 'fldRecoverableGas', 'fldRemainingGas', 'fldInplaceAssGas', 'fldInplaceFreeGas']]

# filtering with user value
df3Filtered = dfHist[dfHist['fldName_x'] == userValue]

# rename columns
df3Filtered = df3Filtered.rename(columns={'fldInplaceOil':'In place Oil','fldRecoverableOil':'Rec Oil','fldRemainingOil':'Remain Oil',
                   'fldRecoverableGas':'Recoverable Gas','fldRemainingGas':'Remaining Gas','fldInplaceAssGas':'Gas in place Ass','fldInplaceFreeGas':'in Place Free Gas'})

# split the filterd dataframe into two dataframes one for oil and for gas
df3FilterdOil = df3Filtered[['In place Oil','Rec Oil','Remain Oil']]
df3FilterdGas = df3Filtered[['Recoverable Gas','Remaining Gas','Gas in place Ass','in Place Free Gas']]

# convert the columns to rows for the bar chart (OIL)
df3FilterdOil_T = df3FilterdOil.T.reset_index()

# selecting the color palette (blue)
color_base = sb.color_palette()[0]

with st.beta_expander('Click here to show histograms',False):
    col1,col2 = st.beta_columns(2)

    sb.barplot(x = 'index',
                y = df3FilterdOil_T[df3FilterdOil_T.columns[1]],
                data = df3FilterdOil_T,
                color=color_base)
    

    plt.title(userValue + ' OIL Distribution');
    plt.xlabel('');
    plt.ylabel(' Oil Volume (MSm3)')

    # Show the plot
    plt.show()
    col1.pyplot()
    
    # convert the columns to rows for the bar chart (GAS)
    df3FilterdGas_T = df3FilterdGas.T.reset_index()

    # selecting the color palette (blue)
    color_base = sb.color_palette()[0]

    sb.barplot(x = 'index',
                y = df3FilterdGas_T[df3FilterdGas_T.columns[1]],
                data = df3FilterdGas_T,
                color = color_base)


    plt.title(userValue + ' GAS Distribution');
    plt.xlabel('');
    plt.ylabel('Gas Volume (BSm3)')

    # Show the plot
    plt.show()
    plt.xticks(fontsize=6)
    col2.pyplot()

#===========================================================================================

def plot_multi2(data,userValues,xtime, cols=None, spacing=.05, **kwargs):


    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return

    del userValues[-1]
    colors = userValues.copy()

    for fluid in userValues:
        if fluid == 'OIL':
            colors[userValues.index('OIL')] = 'green'
        elif fluid == 'GAS':
            colors[userValues.index('GAS')] = 'red'
        elif fluid == 'WATER':
            colors[userValues.index('WATER')] = 'blue'
        elif fluid == 'OE':
            colors[userValues.index('OE')] = 'black'
        elif fluid == 'CONDENSATE':
            colors[userValues.index('CONDENSATE')] = 'orange'
        elif fluid == 'cumulative':
            #colors[userValues.index('cumulative')] = 'magenta'
            colors[userValues.index('cumulative')] = colors[0]

    if xtime == 'yes':
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

    # First axis
    ax = data.loc[:, cols[0]].plot(x_compat=True,label=cols[0], color=colors[0], **kwargs)

    if xtime == 'yes':
        # round to nearest years.
        datemin = np.datetime64(data.index[0], 'Y')
        datemax = np.datetime64(list(data.index)[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
    else:
        # round 
        datemin = 0
        datemax = data.shape[0]
        ax.set_xlim(datemin, datemax)

        ax.tick_params(which='major', width=1)
        ax.tick_params(which='major', length=7)
        plt.xticks(np.arange(0, data.shape[0] +1, 12))
    ax.grid(axis='both', which='both')

    if (cols[0] == 'GAS' or  cols[0] == 'GAS Cumulative Production'):
        ax.set_ylabel(ylabel=cols[0]+ ' (BSm3)')
    else:
        ax.set_ylabel(ylabel=cols[0]+ ' (MSm3)')

    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        if (colors[0]== colors[1]):
            data.loc[:, cols[n]].plot(ax=ax_new,x_compat=True, label=cols[n],linestyle='--', dashes=(5, 10), color=colors[n % len(colors)], **kwargs)
        else:
            data.loc[:, cols[n]].plot(ax=ax_new,x_compat=True, label=cols[n], color=colors[n % len(colors)], **kwargs)

        if (cols[n] == 'GAS'  or  cols[n] == 'GAS Cumulative Production'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' (BSm3)')
        else:
            ax_new.set_ylabel(ylabel=cols[n]+ ' (MSm3)')
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    if xtime == 'yes':
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
    else:
        minor_locator = FixedLocator(data.index.to_list())
        ax.xaxis.set_minor_locator(minor_locator)

    ax.legend(lines, labels, loc=0)
    return ax

def plot_multi3(data,userValues,xtime, cols=None, spacing=.05, **kwargs):


    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return

    del userValues[-1]
    colors = userValues.copy()

    for fluid in userValues:
        if fluid == 'OIL':
            colors[userValues.index('OIL')] = 'green'
        elif fluid == 'GAS':
            colors[userValues.index('GAS')] = 'red'
        elif fluid == 'WATER':
            colors[userValues.index('WATER')] = 'blue'
        elif fluid == 'OE':
            colors[userValues.index('OE')] = 'black'
        elif fluid == 'CONDENSATE':
            colors[userValues.index('CONDENSATE')] = 'orange'
        elif fluid == 'cumulative':
            #colors[userValues.index('cumulative')] = 'magenta'
            colors[userValues.index('cumulative')] = colors[0]

    if xtime == 'yes':
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

    # First axis
    ax = data.loc[:, cols[0]].plot(x_compat=True,label=cols[0], color=colors[0], **kwargs)
    # format the ticks

    # round to nearest years.
    if xtime == 'yes':
        datemin = np.datetime64(data.index[0], 'Y')
        datemax = np.datetime64(list(data.index)[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

    else:
        # round 
        datemin = 0
        datemax = data.shape[0]
        ax.set_xlim(datemin, datemax)

        ax.tick_params(which='major', width=1)
        ax.tick_params(which='major', length=7)
        plt.xticks(np.arange(0, data.shape[0] +1, 12))

    ax.grid(axis='both', which='both')

    if (cols[0] == 'GAS'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production Rate (BSm3/Month)')
    elif (cols[0] == 'GAS Cumulative'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production (BSm3)')
    elif (cols[0] == 'OIL Cumulative'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production (mSm3)')
    elif (cols[0] == 'WATER Cumulative'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production (MSm3)')
    elif (cols[0] == 'OE Cumulative'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production (MSm3)')
    elif (cols[0] == 'CONDENSATE Cumulative'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production (MSm3)')
    else:
        ax.set_ylabel(ylabel=cols[0]+ ' Production Rate (MSm3/Month)')

    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        if (colors[0]== colors[1]):
            data.loc[:, cols[n]].plot(ax=ax_new,x_compat=True, label=cols[n],linestyle='--', dashes=(5, 10), color=colors[n % len(colors)], **kwargs)
        else:
            data.loc[:, cols[n]].plot(ax=ax_new,x_compat=True, label=cols[n], color=colors[n % len(colors)], **kwargs)

        if (cols[n] == 'GAS'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production Rate (BSm3/Month)')
        elif (cols[n] == 'GAS Cumulative'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production (BSm3)')
        elif (cols[n] == 'OIL Cumulative'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production (MSm3)')
        elif (cols[n] == 'WATER Cumulative'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production (MSm3)')
        elif (cols[n] == 'OE Cumulative'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production (MSm3)')
        elif (cols[n] == 'CONDENSATE Cumulative'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production (MSm3)')
        else:
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production Rate (MSm3/Month)')
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label
    if xtime == 'yes':
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
    else:
        minor_locator = FixedLocator(data.index.to_list())
        ax.xaxis.set_minor_locator(minor_locator)

    ax.legend(lines, labels, loc=0)
    return ax

def plot_multi4(data,userValues, cols=None, spacing=.05, **kwargs):


    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return

    #del userValues[-1]
    colors = userValues.copy()

    for fluid in userValues:
        if fluid == 'OIL':
            colors[userValues.index('OIL')] = 'green'
        elif fluid == 'GAS':
            colors[userValues.index('GAS')] = 'red'
        elif fluid == 'WATER':
            colors[userValues.index('WATER')] = 'blue'
        elif fluid == 'OE':
            colors[userValues.index('OE')] = 'black'
        elif fluid == 'CONDENSATE':
            colors[userValues.index('CONDENSATE')] = 'orange'
        elif fluid == 'GOR' or fluid == 'CGR' or fluid == 'WCUT':
            colors[userValues.index(fluid)] = 'purple'

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    # First axis
    ax = data.loc[:, cols[0]].plot(x_compat=True,label=cols[0], color=colors[0], **kwargs)

    # round to nearest years.
    datemin = np.datetime64(data.index[0], 'Y')
    datemax = np.datetime64(list(data.index)[-2], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    ax.grid(axis='both', which='both')


    if (cols[0] == 'GAS'):
        ax.set_ylabel(ylabel=cols[0]+ ' Production Rate (BSm3/Month)')
    elif (cols[0] == 'GOR'):
        ax.set_ylabel(ylabel= 'Gas Oil Ratio (fraction)')
    elif (cols[0] == 'CGR'):
        ax.set_ylabel(ylabel= 'CONDENSATE GAS Ratio (fraction)')
    elif (cols[0] == 'WCUT'):
        ax.set_ylabel(ylabel= 'Water Cut (fraction)')
    else:
        ax.set_ylabel(ylabel=cols[0]+ ' Production Rate (MSm3/Month)')

    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        
        data.loc[:, cols[n]].plot(ax=ax_new,x_compat=True, label=cols[n], color=colors[n % len(colors)], **kwargs)

        if (cols[n] == 'GAS'):
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production Rate (BSm3/Month)')
        elif (cols[n] == 'GOR'):
            ax.set_ylabel(ylabel= 'Gas Oil Ratio (fraction)')
        elif (cols[n] == 'CGR'):
            ax.set_ylabel(ylabel= 'CONDENSATE GAS Ratio (fraction)')
        elif (cols[n] == 'WCUT'):
            ax.set_ylabel(ylabel= 'Water Cut (fraction)')
        else:
            ax_new.set_ylabel(ylabel=cols[n]+ ' Production Rate (MSm3/Month)')
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    ax.legend(lines, labels, loc=0)
    return ax

ans = st.radio('Would you like to indicate a time interval for ploting?',('No','Yes'))
if ans.lower() == 'yes':
    statrtTime = st.text_input("Enter a Start year: ", '1990')
    endTime = st.text_input("Enter an End year: ", '2022')

    statrtTime = int(statrtTime) -1
    statrtTime = str(statrtTime) + '-' + '12'
    endTime = endTime + '-' + '02'
    df_new = df_new[(df_new['Time']> statrtTime) & (df_new['Time']< endTime)]
    dft_new = dft_new[(dft_new.index> statrtTime) & (dft_new.index< endTime)]
    dfMultOil = dfMultOil[(dfMultOil.index> statrtTime) & (dfMultOil.index< endTime)]
    df_new.reset_index(drop=True, inplace=True)


answer = 'both'
yearsx = df_new['Time'].dt.year.to_list()
yearsx = list(set(yearsx))

df_newcSUM = df_new.copy()
for i in range(len(userValues)-1):
    df_newcSUM[userValues[i] + ' Cumulative Production'] = df_newcSUM[userValues[i]].cumsum()

dftt_newcSUM = dft_new.copy()
for i in range(len(userValues)-1):
    dftt_newcSUM[userValues[i] + ' Cumulative Production'] = dftt_newcSUM[userValues[i]].cumsum()

# show table
Numrows = st.text_input("Please select the number of months with recent production you would like to display", '5')
st.text('Last ' + Numrows + ' rows of Filtered Data')
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
st.text('Description Data')
st.dataframe(d)

# creating 5 columns of text to show description
with st.beta_expander('Click here to show full description',False):
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

def plotMult1(df_new,dft_new,xtime):
    # ploting time with Fluid Production
    if graphNum !='1':
        if (answer == 'group' or answer == 'both'):

            if xtime == 'yes':
                years = mdates.YearLocator()   # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                colors2=['green', 'orange','black','blue']
                ax = df_new[mfluids].set_index('Time').plot(figsize=(25,10), color=mcolors,x_compat=True)
                plt.ylabel('Fluid Production Rate (MSm3/Month)');

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

                if 'GAS' in userValues:
                    ax2=ax.twinx()
                    # make a plot with different y-axis using second axis object
                    ax2.plot(dft_new['GAS'],label="GAS",color="red");
                    ax2.set_ylabel("GAS Production Rate (BSm3/Month)")   
                    plt.legend(loc=(0.95,1))

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_minor_locator(months)

                # round to nearest years.
                datemin = np.datetime64(df_new['Time'][0], 'Y') 
                datemax = np.datetime64(list(df_new['Time'])[-2], 'Y') +1
                ax.set_xlim(datemin, datemax)

                ax.tick_params(which='major', width=1)
                ax.tick_params(which='major', length=7)

                ax.grid(axis='both', which='both')

                plt.title(str(userValue)+ ' Field Production' );
                st.pyplot()
            else:

                from matplotlib.ticker import FixedLocator

                colors2=['green', 'orange','black','blue']
                ax = df_new[mfluids].plot(figsize=(25,10), color=mcolors,x_compat=True)
                plt.ylabel('Fluid Production Rate (MSm3/Month)');

                for tick in np.arange(0, df_new.shape[0] +1, 12):
                    plt.axvline(tick,color='black',linewidth=1)

                if 'GAS' in userValues:
                    ax2=ax.twinx()
                    # make a plot with different y-axis using second axis object
                    ax2.plot(dft_new.reset_index()['GAS'],label="GAS",color="red");
                    ax2.set_ylabel("GAS Production Rate (BSm3/Month)")   
                    plt.legend(loc=(0.95,1))

                # format the ticks
                #minor_locator = AutoMinorLocator(2)
                minor_locator = FixedLocator(df_new.index.to_list())
                ax.xaxis.set_minor_locator(minor_locator)

                # round 
                datemin = 0
                datemax = df_new.shape[0]
                ax.set_xlim(datemin, datemax)

                ax.tick_params(which='major', width=1)
                ax.tick_params(which='major', length=7)
                plt.xticks(np.arange(0, df_new.shape[0] +1, 12))

                ax.grid(axis='both', which='both')

                plt.title(str(userValue)+ ' Field Production' );
                st.pyplot()

if st.button('Plot Group Graphs'):
    st.header('Group Graphs')
    #=====================================MultiOil=======================================================
    if ('OIL' in userValues):
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        yearsxoil = dfMultOil.index.year.to_list()
        yearsxoil = list(set(yearsxoil))



        ax = dfMultOil.plot(figsize=(20,10),x_compat=True);

        for year in yearsxoil:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
        plt.title('Oil Production');
        plt.xlabel('Time');
        plt.ylabel('Production Rate (MSm3/Month)');

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(dfMultOil.index[0], 'Y')
        datemax = np.datetime64(list(dfMultOil.index)[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

        ax.grid(axis='both', which='both')
        st.pyplot()

        # months indexes
        dfMultOilShifted = dfMultOil.copy()
        dfMultOilShifted = dfMultOilShifted.apply(lambda x: pd.Series(x.dropna().values))

        # plot
        ax = dfMultOilShifted.reset_index(drop=True).plot(figsize=(20,10),x_compat=True);

        for tick in np.arange(0, dfMultOilShifted.shape[0] +1, 12):
            plt.axvline(tick,color='black',linewidth=1)

        minor_locator = FixedLocator(dfMultOilShifted.reset_index(drop=True).index.to_list())
        ax.xaxis.set_minor_locator(minor_locator)

        plt.title('Oil Production');
        #plt.xlabel('Time');
        plt.ylabel('Production Rate (MSm3/Month)');

        # round
        datemin = 0
        datemax = dfMultOilShifted.shape[0]
        ax.set_xlim(datemin, datemax)

        plt.xticks(np.arange(0, dfMultOilShifted.shape[0] +1, 12))

        ax.grid(axis='both', which='both')
        st.pyplot()
    #============================================================================================


    # ploting time with Fluid Production
    plotMult1(df_new,dft_new,'yes')

    # Trim Oil date Graph
    if ('OIL' in userValues):
        plotMult1(df_new,dft_new,'no')


    def plotMult2(df_newcSUM,dftt_newcSUM,xtime):
        # ploting time with Fluid Production
        if len(graphNum) !=1:
            if (answer == 'group' or answer == 'both'):
                if xtime == 'yes':
                    years = mdates.YearLocator()   # every year
                    months = mdates.MonthLocator()  # every month
                    years_fmt = mdates.DateFormatter('%Y')

                    colors2=['green', 'red', 'orange','black','blue']
                    ax = df_newcSUM[csumNames].set_index('Time').plot(figsize=(25,10), color=mcolors,x_compat=True)
                    plt.ylabel('Fluid Cumulative Production (MSm3)');

                    for year in yearsx:
                        plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

                    if 'GAS' in userValues:
                        ax2=ax.twinx()
                        # make a plot with different y-axis using second axis object
                        ax2.plot(dftt_newcSUM['GAS Cumulative Production'],label="GAS",color="red");
                        ax2.set_ylabel("GAS Cumulative Production (BSm3)")
                        plt.legend(loc=(0.95, 1))
                    

                    # format the ticks
                    ax.xaxis.set_major_locator(years)
                    ax.xaxis.set_major_formatter(years_fmt)
                    ax.xaxis.set_minor_locator(months)

                    # round to nearest years.
                    datemin = np.datetime64(df_newcSUM['Time'][0], 'Y')
                    datemax = np.datetime64(list(df_newcSUM['Time'])[-2], 'Y') + np.timedelta64(1, 'Y')
                    ax.set_xlim(datemin, datemax)

                    ax.grid(axis='both', which='both')

                    plt.title(str(userValue)+ ' Field Cumulative Production' );
                    st.pyplot()

                else:
                    from matplotlib.ticker import FixedLocator

                    colors2=['green', 'orange','black','blue']
                    ax = df_newcSUM[csumNames].plot(figsize=(25,10), color=mcolors,x_compat=True)
                    plt.ylabel('Fluid Cumulative Production (MSm3)');

                    for tick in np.arange(0, df_newcSUM.shape[0] +1, 12):
                        plt.axvline(tick,color='black',linewidth=1)

                    if 'GAS' in userValues:
                        ax2=ax.twinx()
                        # make a plot with different y-axis using second axis object
                        ax2.plot(dftt_newcSUM.reset_index()['GAS Cumulative Production'],label="GAS",color="red");
                        ax2.set_ylabel("GAS Cumulative Production (BSm3)")
                        plt.legend(loc=(0.95, 1))

                    # format the ticks
                    #minor_locator = AutoMinorLocator(2)
                    minor_locator = FixedLocator(df_newcSUM.index.to_list())
                    ax.xaxis.set_minor_locator(minor_locator)

                    # round 
                    datemin = 0
                    datemax = df_newcSUM.shape[0]
                    ax.set_xlim(datemin, datemax)

                    plt.xticks(np.arange(0, df_newcSUM.shape[0] +1, 12))

                    ax.grid(axis='both', which='both')

                    plt.title(str(userValue)+ ' Field Cumulative Production');
                    st.pyplot()

    #  ploting time with Fluid Production
    plotMult2(df_newcSUM,dftt_newcSUM,'yes')

    # Trim Oil date Graph
    if ('OIL' in userValues):
        plotMult2(df_newcSUM,dftt_newcSUM,'no')

    def plotMultiy1(dft_new,xtime):
        userValuesclr = userValues.copy()
        if len(graphNum) !=1:
            if (answer == 'group' or answer == 'both'):
                if xtime == 'yes':
                    for year in yearsx:
                        plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
                else:
                    for tick in np.arange(0, dft_new.shape[0] +1, 12):
                        plt.axvline(tick,color='black',linewidth=1)

                plot_multi3(dft_new,userValuesclr,xtime, figsize=(25, 10));

                plt.title(str(userValue)+ ' Field Production');
                st.pyplot()

    # ploting time with Fluid Production (Multiple y-axis)
    plotMultiy1(dft_new,'yes')

    # ploting time with Fluid Production (Multiple y-axis)(indexes)
    if ('OIL' in userValues):
        plotMultiy1(dft_new.reset_index(drop=True),'no')



    def plotMultiy2(dftt_newcSUM,xtime):
        # ploting time with Fluid Production (Multiple y-axis)
        userValuesclr = userValues.copy()
        if len(graphNum) !=1:
            if (answer == 'group' or answer == 'both'):

                if xtime == 'yes':
                    for year in yearsx:
                        plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
                else:
                    for tick in np.arange(0, dftt_newcSUM.shape[0] +1, 12):
                        plt.axvline(tick,color='black',linewidth=1)

                #df_newcSUM.set_index('Time', inplace=True)
                plot_multi2(dftt_newcSUM,userValuesclr,xtime, figsize=(25, 10));
                
                plt.title(str(userValue)+ ' Field Cumulative Production');
                st.pyplot()

    # ploting time with Fluid Production (Multiple y-axis)
    plotMultiy2(dftt_newcSUM,'yes')

    # ploting time with Fluid Production (Multiple y-axis)(indexes)
    if ('OIL' in userValues):
        plotMultiy2(dftt_newcSUM.reset_index(drop=True),'no')




    #===============================================================================================================================================================#


if (answer == 'individual' or answer =='both' or len(graphNum) ==1):
    lstdf = []
    for i in range(len(userValues)-1):
        dfcSum = df_new.copy()
        dfcSum = dfcSum[[userValues[-1],userValues[i]]]
        dfcSum.set_index('Time', inplace=True)
        dfcSum[userValues[i] + ' Cumulative'] = dfcSum[userValues[i]].cumsum()    
        lstdf.append(dfcSum)
if st.button('Plot Individual Graphs'):
    st.header('Individual Graphs')
    
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

        plot_multi3(lstdf[0],userValuesclr,'yes', figsize=(20, 10));


        if lstdf[0].columns.to_list()[0] == 'GAS':
            plt.title(str(userValue)+ ' Field ' + lstdf[0].columns.to_list()[0] + ' Production');
    
        else:
            plt.title(str(userValue)+ ' Field ' + lstdf[0].columns.to_list()[0] + ' Production');
        st.pyplot()
        

    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=2:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[1]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)


        plot_multi3(lstdf[1],userValuesclr,'yes', figsize=(20, 10));

        if lstdf[1].columns.to_list()[0] == 'GAS':
            plt.title(str(userValue)+ ' Field '  + lstdf[1].columns.to_list()[0] + ' Production');
        else:
            plt.title(str(userValue)+ ' Field ' + lstdf[1].columns.to_list()[0] + ' Production');
        st.pyplot()

    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=3:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[2]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi3(lstdf[2],userValuesclr,'yes', figsize=(20, 10));
        
        if lstdf[2].columns.to_list()[0] == 'GAS':
            plt.title(str(userValue)+ ' Field ' + lstdf[2].columns.to_list()[0] + ' Production');
        else:
            plt.title(str(userValue)+ ' Field ' + lstdf[2].columns.to_list()[0] + ' Production');
        st.pyplot()


    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=4:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[3]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi3(lstdf[3],userValuesclr,'yes', figsize=(20, 10));

        if lstdf[3].columns.to_list()[0] == 'GAS':
            plt.title(str(userValue)+ ' Field ' + lstdf[3].columns.to_list()[0] + ' Production');
        else:
            plt.title(str(userValue)+ ' Field ' + lstdf[3].columns.to_list()[0] + ' Production');
        st.pyplot()


    if (answer == 'individual' or answer == 'both' or len(graphNum) ==1) and len(userValues)-1>=5:
        userValuesclr = userValues.copy()
        #del userValuesclr[-3:]
        userValuesclr[1] = 'cumulative'
        userValuesclr[0] = userValues[4]

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        plot_multi3(lstdf[4],userValuesclr,'yes', figsize=(20, 10));
        
        if lstdf[4].columns.to_list()[0] == 'GAS':
            plt.title(str(userValue)+ ' Field ' + lstdf[4].columns.to_list()[0] + ' Production');
        else:
            plt.title(str(userValue)+ ' Field ' + lstdf[4].columns.to_list()[0] + ' Production');
        st.pyplot()




#===============================================================================================================================================================#
# Calculating GOR and CGR
dfCalc = df_new.copy()
if ('GAS' in userValues) & ('OIL' in userValues):
    dfCalc['GOR'] = ((dfCalc['GAS'])/(dfCalc['OIL']))

if ('GAS' in userValues) & ('CONDENSATE' in userValues):
    dfCalc['CGR'] = ((dfCalc['CONDENSATE'])/(dfCalc['GAS']))

if ('WATER' in userValues) & ('OIL' in userValues):
    dfCalc['WOR'] = ((dfCalc['WATER'])/(dfCalc['OIL']))
    dfCalc['WCUT'] = (((dfCalc['WATER'])/((dfCalc['OIL'])+(dfCalc['WATER']))))

dfCalc = dfCalc.fillna(0)
dfCalc.reset_index(drop=True, inplace=True)
dfCalc2 = dfCalc.copy()
dfCalc2.set_index('Time', inplace=True)


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

        ax = lstdfCalc[index].set_index('Time').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Time'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Time'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

            
        plt.title(str(userValue)+ ' Gas Oil Ratio');
        plt.xlabel('Time');
        plt.ylabel('Gas Oil Ratio (fraction)');
        ax.grid(axis='both', which='both')
        st.pyplot()

    elif ('GAS' in userValues) & ('OIL' in userValues) & (userVal == 'yes'):
        colorscalc = ['GOR','GAS','OIL']
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
            
        plot_multi4(dfCalc2[['GOR','GAS','OIL']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' Gas Oil Ratio');
        plt.xlabel('Time');
        st.pyplot()


    # Plot CGR
    if ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'no'):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'CGR')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Time').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Time'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Time'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

            
        plt.title(str(userValue)+ ' CONDENSATE GAS Ratio');
        plt.xlabel('Time');
        plt.ylabel('CONDENSATE GAS Ratio (fraction)');
        ax.grid(axis='both', which='both')
        st.pyplot()

    elif ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'yes'):
        colorscalc = ['CGR','GAS','CONDENSATE']
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
            
        plot_multi4(dfCalc2[['CGR','GAS','CONDENSATE']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' CONDENSATE GAS Ratio');
        plt.xlabel('Time');
        st.pyplot()


    # Polt WOR
    if ('WATER' in userValues) & ('OIL' in userValues):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'WOR')
    
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Time').plot(figsize=(20,10), color ='purple',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Time'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Time'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

            
        plt.title(str(userValue)+ ' Water Oil Ratio');
        plt.xlabel('Time');
        plt.ylabel('Water Oil Ratio (fraction)');
        ax.grid(axis='both', which='both')
        st.pyplot()


    # Polt WCUT
    if ('WATER' in userValues) & ('OIL' in userValues) & (userVal == 'no'):
        userValuesclr = userValues.copy()
        index = calcIndex(lstdfCalc,'WCUT')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        ax = lstdfCalc[index].set_index('Time').plot(figsize=(20,10), color ='black',x_compat=True);

        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # round to nearest years.
        datemin = np.datetime64(lstdfCalc[index]['Time'][0], 'Y')
        datemax = np.datetime64(list(lstdfCalc[index]['Time'])[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

            
        plt.title(str(userValue)+ ' Water Cut');
        plt.xlabel('Time');
        plt.ylabel('Water Cut (fraction)');
        ax.grid(axis='both', which='both')
        st.pyplot()

    elif ('GAS' in userValues) & ('WATER' in userValues) & (userVal == 'yes'):
        colorscalc = ['WCUT','OIL','WATER',]
        for year in yearsx:
            plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
            
        plot_multi4(dfCalc2[['WCUT','OIL','WATER']],colorscalc, figsize=(20, 10));
        plt.title(str(userValue)+ ' Water Cut');
        plt.xlabel('Time');
        st.pyplot()


# Export data

df_all = pd.merge(df_new,df_newcSUM, on='Time')
df_all = pd.merge(df_all,dfCalc, on='Time')

def DownloadFunc(df):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Data.csv">Download data as csv file</a>'
    return href

st.markdown(DownloadFunc(df_all), unsafe_allow_html=True)
