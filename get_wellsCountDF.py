import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator


class wells:
    
    """
            This function counts the number of cells with specific categories.

            wlbPurpose - main category

            wlbStatus - main category

            Parameters:
            --------------
            Dataframe - df_Wellbore_development (imported from NPD when the application is started)
            - Field names
            - welbore status
            - wellbore names
            - wellbore purpose
            - wellbore content
            - wellbore name category 5 (used to count multilateral wells)

            Returns:
            --------------
            Dataframe:
            df_wells - 
            wellsCountDF - 

            example data:
            - injection wells
            - production wells
            - number of multilateral wells
            - and more

            """

    def get_wellsCountDF(self,df_Wellbore_development,df_Field_Reserves):
        
        # drop columns except ['fldNpdidField','wlbWellboreName','wlbStatus','wlbPurpose','wlbContent','wlbNamePart5']
        df_Wellbore_development = df_Wellbore_development[['fldNpdidField','wlbWellboreName','wlbStatus','wlbPurpose','wlbContent','wlbNamePart5']]

        #df_Wellbore_development.dropna(inplace=True)
        # drop nan values in the id column for mergeing with df_Field_Reserves (otherwise error will occure)
        df_Wellbore_development = df_Wellbore_development[df_Wellbore_development['fldNpdidField'].notna()]

        # convert the id column type to int
        df_Wellbore_development['fldNpdidField'] = df_Wellbore_development['fldNpdidField'].astype(int)

        # drop columns except  ['fldNpdidField','fldName']
        df_Field_Reserves = df_Field_Reserves[['fldNpdidField','fldName']]
    
        # mergeing df_Wellbore_development with df_Field_Reserves (df_wells)
        df_wells = df_Field_Reserves.join(df_Wellbore_development.set_index('fldNpdidField'),on='fldNpdidField', how='left')
        different = list(set(list(df_Wellbore_development['fldNpdidField'].unique())) - set(list(df_Field_Reserves['fldNpdidField'].unique())))
        df_wells = df_wells.append(df_Wellbore_development[df_Wellbore_development['fldNpdidField'].isin(different)],ignore_index=True)
        dfWellsAllFields = df_wells.copy()

        #dfWellsAllFields.head()

        # Y wells multilateral in ['wlbNamePart5'])
        # drop nan values in wlbNamePart5 column to get consistant count of y wells
        # but that won't affect the dfWellsAllFields because we are saving the dropped nan dataframe in a new one (pureY_df)
        pureY_df = dfWellsAllFields[dfWellsAllFields['wlbNamePart5'].notna()]
        # filter the dataframe with rows that only have Y in them
        pureY_df = pureY_df[pureY_df['wlbNamePart5'].str.find('Y') != -1]
        # get the count of Ys in each field in a new df (dfYwellsAllFields)
        dfYwellsAllFields = pureY_df.groupby('fldNpdidField')['wlbNamePart5'].count()
        # making the id and the y count as columns for later merging by reseting the index
        dfYwellsAllFields = dfYwellsAllFields.reset_index()
        # covert the pervious dataframe to a dictionary for later merging
        wells_dict = dfYwellsAllFields.set_index('fldNpdidField')['wlbNamePart5'].to_dict()
        
        #wells_dict

        # PRODUCTION well # Purpose
        # filter  wlbPurpose with value PRODUCTION
        dfProd_PurposeWellsAllFields = dfWellsAllFields[dfWellsAllFields['wlbPurpose'] == 'PRODUCTION']
        # get the count of PRODUCTIONs for each field
        dfProd_PurposeWellsAllFields = dfProd_PurposeWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        # making the id and the PRODUCTIONs count as columns for later merging by reseting the index
        dfProd_PurposeWellsAllFields = dfProd_PurposeWellsAllFields.reset_index()
        # covert the pervious dataframe to a dictionary for later merging
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

        # get all unique fields id and there names from dfWellsAllFields and save them to wellsCountDF
        # which is the df that will be used to merge above counts to
        wellsCountDF = dfWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()
        wellsCountDF = wellsCountDF.reset_index()


        # "groupby". This function ensure that the well count is done for all field names
        wellsCountDF['WellCumSum'] = dfWellsAllFields.groupby('fldNpdidField')['wlbWellboreName'].count()

        #  The high order (.map function) accepts another function and a sequence of "iterables" as parameters...
        #  Return another function.
        # adding Y wells count from wells_dict (which contains the count of Y wells for each field) to wellsCountDF
        # wellsCountDF have field id column, wells_dict also has the field ids
        # so Y wells count will be mapped to wellsCountDF['YwlbWellboreName'] based on the field ids which both have in common
        # it's like merging
        wellsCountDF['YwlbWellboreName'] = wellsCountDF['fldNpdidField'].map(wells_dict)
        # same for others
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

        # for NA and NaN values zero will be filled in the df
        # wellsCountDF before maping above counts to it, it had all field ids and names, it didn't had any nan values
        # when mapping the counts to it, for example the Y wells, the fields who doesn't have Y wells will a value nan
        # but we know that if a filed didn't have a Y well, then it has 0 Y well
        # that's what's the next line is doing fo the Y wells count and other counts
        wellsCountDF = wellsCountDF.fillna(0)
        
        # Spesify datatype integer for all well count categories
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

        # Renaming columns
        wellsCountDF = wellsCountDF.rename(columns={'wlbWellboreName':'well count','YwlbWellboreName':'well count y','wlProdPurposebWellboreName':'production','wlbInjPurposeWellboreName':'injection',
        'wlProdOILbWellboreName':'production oil','wlProdGASbWellboreName':'production gas','wlInjGASbWellboreName':'injection gas','wlInjWATERbWellboreName':'injection water'})


        # NEW CATEGORIES NOT INCLUDED IN THE APPLICATION CODE
        wellsCountDF = wellsCountDF.rename(columns={'wlProdOILANDGASbWellboreName':'production oil/gas','dfProdGASandCondWellsAllFields':'production gas/condensate','wlProdOILANDGASCONDENSATEbWellboreName':'production oil/gas/condensate','wlInjWATERANDGASbWellboreName':'WAG injection',})

        wellsCountDF = wellsCountDF.rename(columns={'wlProdStatusbWellboreName':'production status "active wells"','wlbInjStatusWellboreName':'injection status "active wells"'})
        
        # Drop duplicates
        wellsCountDF = dfWellsAllFields.drop_duplicates(subset='fldNpdidField', keep="last")[['fldNpdidField', 'fldName']].join(wellsCountDF.set_index('fldNpdidField'), on='fldNpdidField', how='left')


        return dfWellsAllFields,df_wells,wellsCountDF



    def plot_multi_oil(self,filterdWells,dfMultOil,uniteType_Oil,final_directory):
        """
                This function plots the oil production vs time(months) and oil production vs time(years).
                Axis grid/ticks are set to every year and every month.

                Parameters:
                --------------
                Dataframe:
                dfMultOil - a dataframe which has oil data for all fields
                uniteType_Oil - a varible which determine the unite of which the graph is plotted with
                final_directory -  a variable that contains the directory where the graphs will be saved for downloading later.
                filterdWells - a dataframe contains the fields which are filtered out by formations by user. those fields will be the ones plotted in Graph1,2

                The Field names used as input data is the fields that are returned based on the users preference in the filter function.
                Dataframe containing all the production data from NPD

                Returns:
                --------------

                list: choosen_filtered_fields is a list of choseen fields from the filtered_fields list,
                this choosen_filtered_fields are used to plot Graph1, Graph2
                Graph 1: oil production vs time (years)
                Graph 2: Oil production vs time (months)

                The two figures are saved in the final_directory as:
                 multiple fields oil rate month
                 multiple fields oil rate year



                """
        # a list of unique fields from filterdWells which the user will choose from
        filtered_fields = list(filterdWells['fldName'].unique())
        # those choosen fields are saved in choosen_filtered_fields and will be used for Graph1,2
        choosen_filtered_fields = st.multiselect('Select wanted fields for plotting', filtered_fields,filtered_fields)
        
        # check if the user selected more tha one field to filter the fields based on them from dfMultOil
        if len(choosen_filtered_fields) > 1:
            dfMultOil = dfMultOil[dfMultOil['Field'].isin(choosen_filtered_fields)]
            
        # check if the user selected one field to filter based on it from dfMultOil
        elif len(choosen_filtered_fields) == 1:
            dfMultOil = dfMultOil[dfMultOil['Field'] == choosen_filtered_fields[0]]
        # nothing is selected, print no data
        else:
            st.text('No data')
        
        # convert rows to columns for plotting
        dfMultOil = dfMultOil.pivot(index='Years', columns='Field',values='prfPrdOilGrossMillSm3')
        
        # plot Graph1, Graph2
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
        return choosen_filtered_fields
