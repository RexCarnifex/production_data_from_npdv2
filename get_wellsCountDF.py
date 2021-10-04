import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator


class wells:

    def get_wellsCountDF(self,df_Wellbore_development,df_Field_Reserves):
        df_Wellbore_development = df_Wellbore_development[['fldNpdidField','wlbWellboreName','wlbStatus','wlbPurpose','wlbContent','wlbNamePart5']]

        #df_Wellbore_development.dropna(inplace=True)
        df_Wellbore_development = df_Wellbore_development[df_Wellbore_development['fldNpdidField'].notna()]

        df_Wellbore_development['fldNpdidField'] = df_Wellbore_development['fldNpdidField'].astype(int)

        df_Field_Reserves = df_Field_Reserves[['fldNpdidField','fldName']]
        ## TODO Change merge to join and inspecting
        #df_wells = pd.merge(df_Wellbore_development,df_Field_Reserves, on='fldNpdidField')
        df_wells = df_Field_Reserves.join(df_Wellbore_development.set_index('fldNpdidField'),on='fldNpdidField', how='left')
        different = list(set(list(df_Wellbore_development['fldNpdidField'].unique())) - set(list(df_Field_Reserves['fldNpdidField'].unique())))
        df_wells = df_wells.append(df_Wellbore_development[df_Wellbore_development['fldNpdidField'].isin(different)],ignore_index=True)
        dfWellsAllFields = df_wells.copy()

        dfWellsAllFields.head()

        # Y wells multilateral in ['wlbNamePart5'])
        pureY_df = dfWellsAllFields[dfWellsAllFields['wlbNamePart5'].notna()]
        pureY_df = pureY_df[pureY_df['wlbNamePart5'].str.find('Y') != -1]
        dfYwellsAllFields = pureY_df.groupby('fldNpdidField')['wlbNamePart5'].count()
        dfYwellsAllFields = dfYwellsAllFields.reset_index()
        wells_dict = dfYwellsAllFields.set_index('fldNpdidField')['wlbNamePart5'].to_dict()

        #wells_dict

        # PRODUCTION well # Purpose
        dfProd_PurposeWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProd_PurposeWellsAllFields = dfProd_PurposeWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProd_PurposeWellsAllFields = dfProd_PurposeWellsAllFields.reset_index()
        wellsProd_dict_Purpose = dfProd_PurposeWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsProd_dict_Purpose

        # PRODUCTION well # Status
        dfProd_StatusWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbStatus'] == 'PRODUCING']
        dfProd_StatusWellsAllFields = dfProd_StatusWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProd_StatusWellsAllFields = dfProd_StatusWellsAllFields.reset_index()
        wellsProd_dict_Status = dfProd_StatusWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsProd_dict_Status

        # INJECTION well # Purpose
        dfInj_PurposeWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'INJECTION']
        dfInj_PurposeWellsAllFields = dfInj_PurposeWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfInj_PurposeWellsAllFields = dfInj_PurposeWellsAllFields.reset_index()
        wellsInj_dict_Purpose = dfInj_PurposeWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsInj_dict_Purpose

        # INJECTION well # Status
        dfInj_StatusWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbStatus'] == 'INJECTING']
        dfInj_StatusWellsAllFields = dfInj_StatusWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfInj_StatusWellsAllFields = dfInj_StatusWellsAllFields.reset_index()
        wellsInj_dict_Status = dfInj_StatusWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsInj_dict_Status

        # PRODUCTION OIL
        dfProdOILWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProdOILWellsAllFields = dfProdOILWellsAllFields[dfProdOILWellsAllFields['wlbContent'] == 'OIL']
        dfProdOILWellsAllFields = dfProdOILWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProdOILWellsAllFields = dfProdOILWellsAllFields.reset_index()
        wellsProdOIL_dict = dfProdOILWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsProdOIL_dict

        # PRODUCTION OIL/GAS
        dfProdOILANDGASWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProdOILANDGASWellsAllFields = dfProdOILANDGASWellsAllFields[dfProdOILANDGASWellsAllFields['wlbContent'] == 'OIL/GAS']
        dfProdOILANDGASWellsAllFields = dfProdOILANDGASWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProdOILANDGASWellsAllFields = dfProdOILANDGASWellsAllFields.reset_index()
        wellsProdOILANDGAS_dict = dfProdOILANDGASWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        # PRODUCTION GAS/CONDENSATE
        dfProdGASandCondWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProdGASandCondWellsAllFields = dfProdGASandCondWellsAllFields[dfProdGASandCondWellsAllFields['wlbContent'] == 'GAS/CONDENSATE']
        dfProdGASandCondWellsAllFields = dfProdGASandCondWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProdGASandCondWellsAllFields = dfProdGASandCondWellsAllFields.reset_index()
        wellsProdGASandCond_dict = dfProdGASandCondWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()



        # PRODUCTION OIL/GASCONDENSATE
        dfProdOILANDGASCONDENSATEWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProdOILANDGASCONDENSATEWellsAllFields = dfProdOILANDGASCONDENSATEWellsAllFields[dfProdOILANDGASCONDENSATEWellsAllFields['wlbContent'] == 'OIL/GAS/CONDENSATE']
        dfProdOILANDGASCONDENSATEWellsAllFields = dfProdOILANDGASCONDENSATEWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProdOILANDGASCONDENSATEWellsAllFields = dfProdOILANDGASCONDENSATEWellsAllFields.reset_index()
        wellsProdOILANDGASCONDENSATE_dict = dfProdOILANDGASCONDENSATEWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsProdOILANDGASCONDENSATE_dict

        # PRODUCTION GAS
        dfProdGASWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        dfProdGASWellsAllFields = dfProdGASWellsAllFields[dfProdGASWellsAllFields['wlbContent'] == 'GAS']
        dfProdGASWellsAllFields = dfProdGASWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfProdGASWellsAllFields = dfProdGASWellsAllFields.reset_index()
        wellsProdGAS_dict = dfProdGASWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsProdGAS_dict

        # INJECTION GAS
        dfInjGASWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'INJECTION']
        dfInjGASWellsAllFields = dfInjGASWellsAllFields[dfInjGASWellsAllFields['wlbContent'] == 'GAS']
        dfInjGASWellsAllFields = dfInjGASWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfInjGASWellsAllFields = dfInjGASWellsAllFields.reset_index()
        wellsInjGAS_dict = dfInjGASWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsInjGAS_dict

        # INJECTION WATER
        dfInjWATERWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'INJECTION']
        dfInjWATERWellsAllFields = dfInjWATERWellsAllFields[dfInjWATERWellsAllFields['wlbContent'] == 'WATER']
        dfInjWATERWellsAllFields = dfInjWATERWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfInjWATERWellsAllFields = dfInjWATERWellsAllFields.reset_index()
        wellsInjWATER_dict = dfInjWATERWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsInjWATER_dict

        # INJECTION WATER/GAS
        dfInjWATERANDGASWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'INJECTION']
        dfInjWATERANDGASWellsAllFields = dfInjWATERANDGASWellsAllFields[dfInjWATERANDGASWellsAllFields['wlbContent'] == 'WATER/GAS']
        dfInjWATERANDGASWellsAllFields = dfInjWATERANDGASWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        dfInjWATERANDGASWellsAllFields = dfInjWATERANDGASWellsAllFields.reset_index()
        wellsInjWATERANDGAS_dict = dfInjWATERANDGASWellsAllFields.set_index('fldNpdidField')['wlbWellboreName'].to_dict()

        #wellsInjWATERANDGAS_dict

        wellsCountDF = dfWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        wellsCountDF = wellsCountDF.reset_index()


        # NEW WELL COUNT SUM
        wellsCountDF['WellCumSum'] = dfWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()


        wellsCountDF['YwlbWellboreName'] = wellsCountDF['fldNpdidField'].map(wells_dict)
        wellsCountDF['wlProdPurposebWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProd_dict_Purpose)
        wellsCountDF['wlbInjPurposeWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsInj_dict_Purpose)
        wellsCountDF['wlProdOILbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProdOIL_dict)
        wellsCountDF['wlProdGASbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProdGAS_dict)
        wellsCountDF['wlInjGASbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsInjGAS_dict)
        wellsCountDF['wlInjWATERbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsInjWATER_dict)


        # NEW CATEGORIES NOT INCLUDED IN THE APPLICATION CODE
        wellsCountDF['wlProdOILANDGASbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProdOILANDGAS_dict)
        wellsCountDF['dfProdGASandCondWellsAllFields'] = wellsCountDF['fldNpdidField'].map(wellsProdGASandCond_dict)

        wellsCountDF['wlProdOILANDGASCONDENSATEbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProdOILANDGASCONDENSATE_dict)
        wellsCountDF['wlInjWATERANDGASbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsInjWATERANDGAS_dict)

        wellsCountDF['wlProdStatusbWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsProd_dict_Status)
        wellsCountDF['wlbInjStatusWellboreName'] = wellsCountDF['fldNpdidField'].map(wellsInj_dict_Status)


        wellsCountDF = wellsCountDF.fillna(0)
        wellsCountDF['YwlbWellboreName'] = wellsCountDF['YwlbWellboreName'].astype(int)
        wellsCountDF['wlProdPurposebWellboreName'] = wellsCountDF['wlProdPurposebWellboreName'].astype(int)
        wellsCountDF['wlbInjPurposeWellboreName'] = wellsCountDF['wlbInjPurposeWellboreName'].astype(int)
        wellsCountDF['wlProdOILbWellboreName'] = wellsCountDF['wlProdOILbWellboreName'].astype(int)
        wellsCountDF['wlProdGASbWellboreName'] = wellsCountDF['wlProdGASbWellboreName'].astype(int)
        wellsCountDF['wlInjGASbWellboreName'] = wellsCountDF['wlInjGASbWellboreName'].astype(int)
        wellsCountDF['wlInjWATERbWellboreName'] = wellsCountDF['wlInjWATERbWellboreName'].astype(int)


        # NEW CATEGORIES NOT INCLUDED IN THE APPLICATION CODE
        wellsCountDF['wlProdOILANDGASbWellboreName'] = wellsCountDF['wlProdOILANDGASbWellboreName'].astype(int)
        wellsCountDF['dfProdGASandCondWellsAllFields'] = wellsCountDF['dfProdGASandCondWellsAllFields'].astype(int)

        wellsCountDF['wlProdOILANDGASCONDENSATEbWellboreName'] = wellsCountDF['wlProdOILANDGASCONDENSATEbWellboreName'].astype(int)
        wellsCountDF['wlInjWATERANDGASbWellboreName'] = wellsCountDF['wlInjWATERANDGASbWellboreName'].astype(int)

        wellsCountDF['wlProdStatusbWellboreName'] = wellsCountDF['wlProdStatusbWellboreName'].astype(int)
        wellsCountDF['wlbInjStatusWellboreName'] = wellsCountDF['wlbInjStatusWellboreName'].astype(int)

        wellsCountDF = wellsCountDF.rename(columns={'wlbWellboreName':'well count','YwlbWellboreName':'well count y','wlProdPurposebWellboreName':'production','wlbInjPurposeWellboreName':'injection',
        'wlProdOILbWellboreName':'production oil','wlProdGASbWellboreName':'production gas','wlInjGASbWellboreName':'injection gas','wlInjWATERbWellboreName':'injection water'})


        # NEW CATEGORIES NOT INCLUDED IN THE APPLICATION CODE
        wellsCountDF = wellsCountDF.rename(columns={'wlProdOILANDGASbWellboreName':'production oil/gas','dfProdGASandCondWellsAllFields':'production gas/condensate','wlProdOILANDGASCONDENSATEbWellboreName':'production oil/gas/condensate','wlInjWATERANDGASbWellboreName':'WAG injection',})

        wellsCountDF = wellsCountDF.rename(columns={'wlProdStatusbWellboreName':'production status "active wells"','wlbInjStatusWellboreName':'injection status "active wells"',})
        wellsCountDF = dfWellsAllFields.drop_duplicates(subset='fldNpdidField', keep="last")[['fldNpdidField', 'fldName']].join(wellsCountDF.set_index('fldNpdidField'), on='fldNpdidField', how='left')


        return dfWellsAllFields,df_wells,wellsCountDF



    def plot_multi_oil(self,filterdWells,dfMultOil,uniteType_Oil,final_directory):
        filtered_fields = list(filterdWells['fldName'].unique())
        choosen_filtered_fields = st.multiselect('Select wanted fields for plotting', filtered_fields,filtered_fields)
        
        if len(choosen_filtered_fields) > 1:
            dfMultOil = dfMultOil[dfMultOil['Field'].isin(choosen_filtered_fields)]
        elif len(choosen_filtered_fields) == 1:
            dfMultOil = dfMultOil[dfMultOil['Field'] == choosen_filtered_fields[0]]
        else:
            st.text('No data')

        dfMultOil = dfMultOil.pivot(index='Years', columns='Field',values='prfPrdOilGrossMillSm3')
        if len(choosen_filtered_fields) >= 1:
            if st.button('Plot Multi Oil graph for filtered fields from formations'):
                if uniteType_Oil == 'STB':
                    dfMultOil = dfMultOil*6.2898

                years = mdates.YearLocator()   # every year
                months = mdates.MonthLocator()  # every month
                years_fmt = mdates.DateFormatter('%Y')

                yearsxoil = dfMultOil.index.year.to_list()
                yearsxoil = list(set(yearsxoil))



                ax = dfMultOil.plot(figsize=(20,10),x_compat=True);

                for year in yearsxoil:
                    plt.axvline(pd.Timestamp(str(year)),color='black',linewidth=1)
                plt.title('Oil Production');
                plt.xlabel('Years');
                if uniteType_Oil == 'STB':
                    plt.ylabel('Production Rate (STB/Month)');
                else:
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
                plt.savefig(final_directory + '/' + ' multiple fields oil rate year.png')
                st.pyplot()

                # months indexes
                dfMultOilShifted = dfMultOil.copy()
                dfMultOilShifted = dfMultOilShifted.apply(lambda x: pd.Series(x.dropna().values))

                # plot
                ax = dfMultOilShifted.reset_index(drop=True).plot(figsize=(20,10),x_compat=True);
                plt.xlabel('Months');

                for tick in np.arange(0, dfMultOilShifted.shape[0] +1, 12):
                    plt.axvline(tick,color='black',linewidth=1)

                minor_locator = FixedLocator(dfMultOilShifted.reset_index(drop=True).index.to_list())
                ax.xaxis.set_minor_locator(minor_locator)

                plt.title('Oil Production');
                #plt.xlabel('Time');
                if uniteType_Oil == 'STB':
                    plt.ylabel('Production Rate (STB/Month)');
                else:
                    plt.ylabel('Production Rate (MSm3/Month)');

                # round
                datemin = 0
                datemax = dfMultOilShifted.shape[0]
                ax.set_xlim(datemin, datemax)

                plt.xticks(np.arange(0, dfMultOilShifted.shape[0] +1, 12))

                ax.grid(axis='both', which='both')
                plt.savefig(final_directory + '/' + ' multiple fields oil rate month.png')
                st.pyplot()
