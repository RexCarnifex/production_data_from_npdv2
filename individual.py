import shutil
import pandas as pd
import numpy as np
import base64
import zipfile
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import streamlit as st
import os

from plot_multi_helper import plot_multi_helper


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


class Indivdual:
    def histogram(self,df1Hist,df2Hist,userValue,uniteType_Oil,final_directory,uniteType_Gas):
        # mergeng the two dataframes
        dfHist = pd.merge(df1Hist, df2Hist, on='fldNpdidField')

        # Extracting needed columns for charts
        dfHist = dfHist[['fldName_x', 'fldInplaceOil', 'fldRecoverableOil', 'fldRemainingOil', 'fldRecoverableGas',
                         'fldRemainingGas', 'fldInplaceAssGas', 'fldInplaceFreeGas']]

        # filtering with user value
        df3Filtered = dfHist[dfHist['fldName_x'] == userValue]

        # rename columns
        df3Filtered = df3Filtered.rename(
            columns={'fldInplaceOil': 'OIIP', 'fldRecoverableOil': 'EUR_oil', 'fldRemainingOil': 'NPD TRR_oil',
                     'fldRecoverableGas': 'EUR_gas', 'fldRemainingGas': 'NPD TRR_gas', 'fldInplaceAssGas': 'GIIP_ass',
                     'fldInplaceFreeGas': 'GIIP_free'})

        # split the filterd dataframe into two dataframes one for oil and for gas
        df3FilterdOil = df3Filtered[['OIIP', 'EUR_oil', 'NPD TRR_oil']]
        df3FilterdGas = df3Filtered[['GIIP_free', 'GIIP_ass', 'EUR_gas', 'NPD TRR_gas']]

        if uniteType_Oil == 'STB':
            df3FilterdOil = df3FilterdOil * 6.2898

        # convert the columns to rows for the bar chart (OIL)
        df3FilterdOil_T = df3FilterdOil.T.reset_index()

        # selecting the color palette (green)
        color_base = sb.color_palette()[2]

        with st.beta_expander('Display/hide histograms', True):
            col1, col2 = st.beta_columns(2)

            ax = sb.barplot(x='index',
                            y=df3FilterdOil_T[df3FilterdOil_T.columns[1]],
                            data=df3FilterdOil_T,
                            color=color_base)
            ax.bar_label(ax.containers[0], fmt='%.2f');

            plt.title(userValue + ' Oil Volumes');
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
                df3FilterdGas = df3FilterdGas * 35.315

            # convert the columns to rows for the bar chart (GAS)
            df3FilterdGas_T = df3FilterdGas.T.reset_index()

            # selecting the color palette (blue)
            color_base = sb.color_palette()[3]

            ax = sb.barplot(x='index',
                            y=df3FilterdGas_T[df3FilterdGas_T.columns[1]],
                            data=df3FilterdGas_T,
                            color=color_base)

            ax.bar_label(ax.containers[0], fmt='%.2f');

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

    def wells_status_histograms(self,df_Wellbore_development,df_Field_Reserves,userValue,final_directory):
        with st.beta_expander("Display well's status histograms", False):
            from get_wellsCountDF import wells
            dfWellsAllFields, df_wells, wellsCountDF = wells().get_wellsCountDF(df_Wellbore_development,
                                                                                df_Field_Reserves)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            df_wells = df_wells[df_wells['fldName'] == userValue].set_index(['fldName'])

            st.text("Well's field information. The total number of wells:" + str(df_wells['wlbWellboreName'].nunique()))
            # Num of wells name that have Y in it
            pureY_df = df_wells[df_wells['wlbNamePart5'].notna()]
            df_wellsY = pureY_df[pureY_df['wlbNamePart5'].str.find('Y') != -1]
            yCount = df_wellsY.shape[0]
            st.text("Number of wells planned as multilateral wellbores (Y):" + str(yCount))

            # dropdown status selecttion
            stlst = df_wells['wlbStatus'].unique()
            stselc = st.selectbox('Select a status to filtter with', stlst)
            st.dataframe(df_wells[df_wells['wlbStatus'] == stselc])
            # save the well count df with the main well
            # --------------------------------------------------------------
            final_directory_excel_wells = os.path.join(os.getcwd(), r'wells excel')
            if os.path.exists(final_directory_excel_wells):
                shutil.rmtree(final_directory_excel_wells)
            if not os.path.exists(final_directory_excel_wells):
                os.makedirs(final_directory_excel_wells)

            from pandas import ExcelWriter
            w = ExcelWriter(final_directory_excel_wells + '/' + 'Wells.xlsx')
            dfWellsAllFields.to_excel(w, sheet_name='Sheet0', index=False)
            wellsCountDF.to_excel(w, sheet_name='Sheet1', index=False)
            w.save()

            st.markdown(get_binary_file_downloader_html(final_directory_excel_wells + '/' + 'Wells.xlsx', 'Well count and the main well data'),unsafe_allow_html=True)
            # --------------------------------------------------------------
            from plot_wells_status_purpose import plot_wells_status_purpose_content
            plot_wells_status_purpose_content().plot_contetnt(df_wells, userValue, final_directory)
            col1, col2 = st.beta_columns(2)
            plot_wells_status_purpose_content().plot_status(df_wells, userValue, final_directory, col1)
            plot_wells_status_purpose_content().plot_purpose(df_wells, userValue, final_directory, col2)

    def comulative_calcuations(self,answer,df_new,uniteType_Oil,userValues,uniteType_Gas,graphNum):
        if (answer == 'individual' or answer == 'both' or len(graphNum) == 1):
            lstdf = []
            for i in range(len(userValues) - 1):
                dfcSum = df_new.copy()
                if ('OIL' in userValues) & (uniteType_Oil == 'STB'):
                    dfcSum['OIL'] = dfcSum['OIL'] * 6.2898
                if ('GAS' in userValues) & (uniteType_Gas == 'ft3'):
                    dfcSum['GAS'] = dfcSum['GAS'] * 35.315
                dfcSum = dfcSum[[userValues[-1], userValues[i]]]
                dfcSum.set_index('Years', inplace=True)
                dfcSum[userValues[i] + ' Cumulative'] = dfcSum[userValues[i]].cumsum()
                lstdf.append(dfcSum)

            return lstdf

    def plot_indivdual(self,answer,graphNum,userValues,yearsx,uniteType_Gas,uniteType_Oil,lstdf,userValue,final_directory2):
        if st.button('Plot Individual Graphs'):
            st.header('Individual Graphs')
            groupORindiv = 'indiv'

            if (answer == 'individual' or answer == 'both' or len(graphNum) == 1) and (len(userValues) - 1 >= 1):
                userValuesclr = userValues.copy()
                # del userValuesclr[-3:]
                if len(graphNum) == 1:
                    userValuesclr.append('cumu')
                    userValuesclr[1] = 'cumulative'
                else:
                    userValuesclr[1] = 'cumulative'

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi3(groupORindiv, uniteType_Gas, uniteType_Oil, lstdf[0], userValuesclr,
                                                'yes', figsize=(20, 10));

                plt.title(str(userValue) + ' Field ' + lstdf[0].columns.to_list()[0] + ' Production');

                plt.savefig(
                    final_directory2 + '/' + userValue + ' Field ' + lstdf[0].columns.to_list()[0] + ' Production.png')
                st.pyplot()

            if (answer == 'individual' or answer == 'both' or len(graphNum) == 1) and len(userValues) - 1 >= 2:
                userValuesclr = userValues.copy()
                # del userValuesclr[-3:]
                userValuesclr[1] = 'cumulative'
                userValuesclr[0] = userValues[1]

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi3(groupORindiv, uniteType_Gas, uniteType_Oil, lstdf[1], userValuesclr,
                                                'yes', figsize=(20, 10));

                plt.title(str(userValue) + ' Field ' + lstdf[1].columns.to_list()[0] + ' Production');

                plt.savefig(
                    final_directory2 + '/' + userValue + ' Field ' + lstdf[1].columns.to_list()[0] + ' Production.png')
                st.pyplot()

            if (answer == 'individual' or answer == 'both' or len(graphNum) == 1) and len(userValues) - 1 >= 3:
                userValuesclr = userValues.copy()
                # del userValuesclr[-3:]
                userValuesclr[1] = 'cumulative'
                userValuesclr[0] = userValues[2]

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi3(groupORindiv, uniteType_Gas, uniteType_Oil, lstdf[2], userValuesclr,
                                                'yes', figsize=(20, 10));

                plt.title(str(userValue) + ' Field ' + lstdf[2].columns.to_list()[0] + ' Production');

                plt.savefig(
                    final_directory2 + '/' + userValue + ' Field ' + lstdf[2].columns.to_list()[0] + ' Production.png')
                st.pyplot()

            if (answer == 'individual' or answer == 'both' or len(graphNum) == 1) and len(userValues) - 1 >= 4:
                userValuesclr = userValues.copy()
                # del userValuesclr[-3:]
                userValuesclr[1] = 'cumulative'
                userValuesclr[0] = userValues[3]

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi3(groupORindiv, uniteType_Gas, uniteType_Oil, lstdf[3], userValuesclr,
                                                'yes', figsize=(20, 10));

                plt.title(str(userValue) + ' Field ' + lstdf[3].columns.to_list()[0] + ' Production');

                plt.savefig(
                    final_directory2 + '/' + userValue + ' Field ' + lstdf[3].columns.to_list()[0] + ' Production.png')
                st.pyplot()

            if (answer == 'individual' or answer == 'both' or len(graphNum) == 1) and len(userValues) - 1 >= 5:
                userValuesclr = userValues.copy()
                # del userValuesclr[-3:]
                userValuesclr[1] = 'cumulative'
                userValuesclr[0] = userValues[4]

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi3(groupORindiv, uniteType_Gas, uniteType_Oil, lstdf[4], userValuesclr,
                                                'yes', figsize=(20, 10));

                plt.title(str(userValue) + ' Field ' + lstdf[4].columns.to_list()[0] + ' Production');

                plt.savefig(
                    final_directory2 + '/' + userValue + ' Field ' + lstdf[4].columns.to_list()[0] + ' Production.png')
                st.pyplot()

            # create plots download link
            zipf = zipfile.ZipFile('individual Plots.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('individual Plots', zipf)
            zipf.close()
            st.markdown(get_binary_file_downloader_html('individual Plots.zip', userValue + ' individual Plots'),
                        unsafe_allow_html=True)

    def get_dfCalc(self,userValues,df_new):
        dfCalc = df_new.copy()
        if ('GAS' in userValues) & ('OIL' in userValues):
            dfCalc['GOR'] = ((dfCalc['GAS'] * 1000) / (dfCalc['OIL']))

        if ('GAS' in userValues) & ('CONDENSATE' in userValues):
            dfCalc['CGR'] = ((dfCalc['CONDENSATE']) / (dfCalc['GAS'] * 1000))

        if ('WATER' in userValues) & ('OIL' in userValues):
            dfCalc['WOR'] = ((dfCalc['WATER']) / (dfCalc['OIL']))
            dfCalc['WCUT'] = (((dfCalc['WATER']) / ((dfCalc['OIL']) + (dfCalc['WATER']))))

        dfCalc = dfCalc.fillna(0)
        dfCalc.reset_index(drop=True, inplace=True)
        dfCalc2 = dfCalc.copy()
        dfCalc2.set_index('Years', inplace=True)

        fluids = userValues.copy()

        del fluids[-1]

        dfCalc = dfCalc.drop(columns=fluids)
        return dfCalc,dfCalc2

    def calculations_GOR_CGR_plot(self,df_new,userValues,yearsx,userValue,final_directory3):

        dfCalc,dfCalc2 = self.get_dfCalc(userValues,df_new)
        columnNamesCalc = dfCalc.columns.to_list()

        lstdfCalc = []
        for i in range(len(columnNamesCalc) - 1):
            lstdfCalc.append(dfCalc[[columnNamesCalc[0], columnNamesCalc[i + 1]]])

        def calcIndex(lstdfCalc, name):
            for i in range(4):
                if lstdfCalc[i][lstdfCalc[i].columns[1]].name == name:
                    return i

        userVal = st.radio('Do you want to plot the GOR/CGR/WCUT with the fluids rates?', ('Yes', 'No'))
        userVal = userVal.lower()

        # ===============================================================================================================================================================#
        # Ploting
        if st.button('Plot Calculations Graphs'):
            st.header('Calculations Graphs')

            # Plot GOR
            if ('GAS' in userValues) & ('OIL' in userValues) & (userVal == 'no'):
                userValuesclr = userValues.copy()
                index = calcIndex(lstdfCalc, 'GOR')

                years = mdates.YearLocator()  # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                ax = lstdfCalc[index].set_index('Years').plot(figsize=(20, 10), color='black', x_compat=True);

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_minor_locator(months)

                # round to nearest years.
                datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
                datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
                ax.set_xlim(datemin, datemax)

                plt.title(str(userValue) + ' Gas Oil Ratio');
                plt.xlabel('Years');
                plt.ylabel('Gas Oil Ratio (fraction)');
                ax.grid(axis='both', which='both')
                plt.savefig(final_directory3 + '/' + str(userValue) + ' Gas Oil Ratio.png')
                st.pyplot()

            elif ('GAS' in userValues) & ('OIL' in userValues) & (userVal == 'yes'):
                colorscalc = ['GOR', 'GAS', 'OIL']
                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi4(dfCalc2[['GOR', 'GAS', 'OIL']], colorscalc, figsize=(20, 10));
                plt.title(str(userValue) + ' Gas Oil Ratio');
                plt.xlabel('Years');
                plt.savefig(final_directory3 + '/' + str(userValue) + ' Gas Oil Ratio.png')
                st.pyplot()

            # Plot CGR
            if ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'no'):
                userValuesclr = userValues.copy()
                index = calcIndex(lstdfCalc, 'CGR')

                years = mdates.YearLocator()  # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                ax = lstdfCalc[index].set_index('Years').plot(figsize=(20, 10), color='black', x_compat=True);

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_minor_locator(months)

                # round to nearest years.
                datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
                datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
                ax.set_xlim(datemin, datemax)

                plt.title(str(userValue) + ' CONDENSATE GAS Ratio');
                plt.xlabel('Years');
                plt.ylabel('CONDENSATE GAS Ratio (fraction)');
                ax.grid(axis='both', which='both')
                plt.savefig(final_directory3 + '/' + str(userValue) + ' CONDENSATE GAS Ratio.png')
                st.pyplot()

            elif ('GAS' in userValues) & ('CONDENSATE' in userValues) & (userVal == 'yes'):
                colorscalc = ['CGR', 'GAS', 'CONDENSATE']
                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi4(dfCalc2[['CGR', 'GAS', 'CONDENSATE']], colorscalc, figsize=(20, 10));
                plt.title(str(userValue) + ' CONDENSATE GAS Ratio');
                plt.xlabel('Years');
                plt.savefig(final_directory3 + '/' + str(userValue) + ' CONDENSATE GAS Ratio.png')
                st.pyplot()

            # Polt WOR
            if ('WATER' in userValues) & ('OIL' in userValues):
                userValuesclr = userValues.copy()
                index = calcIndex(lstdfCalc, 'WOR')

                years = mdates.YearLocator()  # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                ax = lstdfCalc[index].set_index('Years').plot(figsize=(20, 10), color='purple', x_compat=True);

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_minor_locator(months)

                # round to nearest years.
                datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
                datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
                ax.set_xlim(datemin, datemax)

                plt.title(str(userValue) + ' Water Oil Ratio');
                plt.xlabel('Years');
                plt.ylabel('Water Oil Ratio (fraction)');
                ax.grid(axis='both', which='both')
                plt.savefig(final_directory3 + '/' + str(userValue) + ' Water Oil Ratio.png')
                st.pyplot()

            # Polt WCUT
            if ('WATER' in userValues) & ('OIL' in userValues) & (userVal == 'no'):
                userValuesclr = userValues.copy()
                index = calcIndex(lstdfCalc, 'WCUT')

                years = mdates.YearLocator()  # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                ax = lstdfCalc[index].set_index('Years').plot(figsize=(20, 10), color='black', x_compat=True);

                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                # format the ticks
                ax.xaxis.set_major_locator(years)
                ax.xaxis.set_major_formatter(years_fmt)
                ax.xaxis.set_minor_locator(months)

                # round to nearest years.
                datemin = np.datetime64(lstdfCalc[index]['Years'][0], 'Y')
                datemax = np.datetime64(list(lstdfCalc[index]['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
                ax.set_xlim(datemin, datemax)

                plt.title(str(userValue) + ' Water Cut');
                plt.xlabel('Years');
                plt.ylabel('Water Cut (fraction)');
                ax.grid(axis='both', which='both')
                plt.savefig(final_directory3 + '/' + str(userValue) + ' Water Cut.png')
                st.pyplot()

            elif ('GAS' in userValues) & ('WATER' in userValues) & (userVal == 'yes'):
                colorscalc = ['WCUT', 'OIL', 'WATER', ]
                for year in yearsx:
                    plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                plot_multi_helper().plot_multi4(dfCalc2[['WCUT', 'OIL', 'WATER']], colorscalc, figsize=(20, 10));
                plt.title(str(userValue) + ' Water Cut');
                plt.xlabel('Years');
                plt.savefig(final_directory3 + '/' + str(userValue) + ' Water Cut.png')
                st.pyplot()

            # create plots download link
            zipf = zipfile.ZipFile('Calculation Plots.zip', 'w', zipfile.ZIP_DEFLATED)
            zipdir('Calculation Plots', zipf)
            zipf.close()

            st.markdown(get_binary_file_downloader_html('Calculation Plots.zip', userValue + ' Calculation Plots'),
                        unsafe_allow_html=True)




