import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import streamlit as st
class plot_multi_helper:
    def plotMult1(self,df_new, dft_new, xtime,graphNum,answer,mfluids,yearsx,userValues,userValue,final_directory,mcolors):
        # ploting time with Fluid Production
        if graphNum != '1':
            if (answer == 'group' or answer == 'both'):

                if xtime == 'yes':
                    years = mdates.YearLocator()  # every year
                    months = mdates.MonthLocator()  # every month
                    years_fmt = mdates.DateFormatter('%Y')

                    colors2 = ['green', 'orange', 'black', 'blue']
                    ax = df_new[mfluids].set_index('Years').plot(figsize=(25, 10), color=mcolors, x_compat=True)
                    plt.ylabel('Fluid Production Rate (MSm3/Month)');

                    for year in yearsx:
                        plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                    if 'GAS' in userValues:
                        ax2 = ax.twinx()
                        # make a plot with different y-axis using second axis object
                        ax2.plot(dft_new['GAS'], label="GAS", color="red");
                        ax2.set_ylabel("GAS Production Rate (BSm3/Month)")
                        plt.legend(loc=(0.95, 1))

                    # format the ticks
                    ax.xaxis.set_major_locator(years)
                    ax.xaxis.set_major_formatter(years_fmt)
                    ax.xaxis.set_minor_locator(months)

                    # round to nearest years.
                    datemin = np.datetime64(df_new['Years'][0], 'Y')
                    datemax = np.datetime64(list(df_new['Years'])[-2], 'Y') + 1
                    ax.set_xlim(datemin, datemax)

                    ax.tick_params(which='major', width=1)
                    ax.tick_params(which='major', length=7)

                    ax.grid(axis='both', which='both')

                    plt.title(str(userValue) + ' Field Production');
                    plt.savefig(final_directory + '/' + userValue + ' field production year.png')
                    st.pyplot()
                else:

                    from matplotlib.ticker import FixedLocator

                    colors2 = ['green', 'orange', 'black', 'blue']
                    ax = df_new[mfluids].plot(figsize=(25, 10), color=mcolors, x_compat=True)
                    plt.ylabel('Fluid Production Rate (MSm3/Month)');
                    plt.xlabel('Months');

                    for tick in np.arange(0, df_new.shape[0] + 1, 12):
                        plt.axvline(tick, color='black', linewidth=1)

                    if 'GAS' in userValues:
                        ax2 = ax.twinx()
                        # make a plot with different y-axis using second axis object
                        ax2.plot(dft_new.reset_index()['GAS'], label="GAS", color="red");
                        ax2.set_ylabel("GAS Production Rate (BSm3/Month)")
                        plt.legend(loc=(0.95, 1))

                    # format the ticks
                    # minor_locator = AutoMinorLocator(2)
                    minor_locator = FixedLocator(df_new.index.to_list())
                    ax.xaxis.set_minor_locator(minor_locator)

                    # round
                    datemin = 0
                    datemax = df_new.shape[0]
                    ax.set_xlim(datemin, datemax)

                    ax.tick_params(which='major', width=1)
                    ax.tick_params(which='major', length=7)
                    plt.xticks(np.arange(0, df_new.shape[0] + 1, 12))

                    ax.grid(axis='both', which='both')

                    plt.title(str(userValue) + ' Field Production');
                    plt.xlabel('Months of production')
                    plt.savefig(final_directory + '/' + userValue + ' field production month.png')
                    st.pyplot()


    def plot_multi2(self,data, userValues, xtime, cols=None, spacing=.05, **kwargs):

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
                # colors[userValues.index('cumulative')] = 'magenta'
                colors[userValues.index('cumulative')] = colors[0]

        if xtime == 'yes':
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')

        # First axis
        ax = data.loc[:, cols[0]].plot(x_compat=True, label=cols[0], color=colors[0], **kwargs)
        if xtime == 'no':
            plt.xlabel('Months');

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
            plt.xticks(np.arange(0, data.shape[0] + 1, 12))
        ax.grid(axis='both', which='both')

        if (cols[0] == 'GAS' or cols[0] == 'GAS Cumulative Production'):
            ax.set_ylabel(ylabel=cols[0] + ' (BSm3)')
        else:
            ax.set_ylabel(ylabel=cols[0] + ' (MSm3)')

        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            if (colors[0] == colors[1]):
                data.loc[:, cols[n]].plot(ax=ax_new, x_compat=True, label=cols[n], linestyle='--', dashes=(5, 10),
                                          color=colors[n % len(colors)], **kwargs)
            else:
                data.loc[:, cols[n]].plot(ax=ax_new, x_compat=True, label=cols[n], color=colors[n % len(colors)],
                                          **kwargs)

            if (cols[n] == 'GAS' or cols[n] == 'GAS Cumulative Production'):
                ax_new.set_ylabel(ylabel=cols[n] + ' (BSm3)')
            else:
                ax_new.set_ylabel(ylabel=cols[n] + ' (MSm3)')

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

    def plot_multi3(self,groupORindiv,uniteType_Gas,uniteType_Oil,data, userValues, xtime, cols=None, spacing=.05, **kwargs):

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
                # colors[userValues.index('cumulative')] = 'magenta'
                colors[userValues.index('cumulative')] = colors[0]

        if xtime == 'yes':
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')

        # First axis
        ax = data.loc[:, cols[0]].plot(x_compat=True, label=cols[0], color=colors[0], **kwargs)
        if xtime == 'no':
            plt.xlabel('Months');
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
            plt.xticks(np.arange(0, data.shape[0] + 1, 12))

        ax.grid(axis='both', which='both')

        if groupORindiv == 'indiv':
            if (cols[0] == 'GAS'):
                if uniteType_Gas == 'ft3':
                    ax.set_ylabel(ylabel=cols[0] + ' Production Rate (Bft3/Month)')
                else:
                    ax.set_ylabel(ylabel=cols[0] + ' Production Rate (BSm3/Month)')
            elif (cols[0] == 'GAS Cumulative'):
                if uniteType_Gas == 'ft3':
                    ax.set_ylabel(ylabel=cols[0] + ' Production (Bft3)')
                else:
                    ax.set_ylabel(ylabel=cols[0] + ' Production (BSm3)')
            elif (cols[0] == 'OIL Cumulative'):
                if uniteType_Oil == 'STB':
                    ax.set_ylabel(ylabel=cols[0] + ' Production (MSTB)')
                else:
                    ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'OIL'):
                if uniteType_Oil == 'STB':
                    ax.set_ylabel(ylabel=cols[0] + ' Production Rate (MSTB/Month)')
                else:
                    ax.set_ylabel(ylabel=cols[0] + ' Production Rate (MSm3/Month)')
            elif (cols[0] == 'WATER Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'OE Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'CONDENSATE Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            else:
                ax.set_ylabel(ylabel=cols[0] + ' Production Rate (MSm3/Month)')

        else:
            if (cols[0] == 'GAS'):
                ax.set_ylabel(ylabel=cols[0] + ' Production Rate (BSm3/Month)')
            elif (cols[0] == 'GAS Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (BSm3)')
            elif (cols[0] == 'OIL Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'WATER Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'OE Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            elif (cols[0] == 'CONDENSATE Cumulative'):
                ax.set_ylabel(ylabel=cols[0] + ' Production (MSm3)')
            else:
                ax.set_ylabel(ylabel=cols[0] + ' Production Rate (MSm3/Month)')

        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            if (colors[0] == colors[1]):
                data.loc[:, cols[n]].plot(ax=ax_new, x_compat=True, label=cols[n], linestyle='--', dashes=(5, 10),
                                          color=colors[n % len(colors)], **kwargs)
            else:
                data.loc[:, cols[n]].plot(ax=ax_new, x_compat=True, label=cols[n], color=colors[n % len(colors)],
                                          **kwargs)

            if groupORindiv == 'indiv':
                if (cols[n] == 'GAS'):
                    if uniteType_Gas == 'ft3':
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (Bft3/Month)')
                    else:
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (BSm3/Month)')
                elif (cols[n] == 'GAS Cumulative'):
                    if uniteType_Gas == 'ft3':
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production (Bft3)')
                    else:
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production (BSm3)')
                elif (cols[n] == 'OIL Cumulative'):
                    if uniteType_Oil == 'STB':
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production (STB)')
                    else:
                        ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'WATER Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'OE Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'CONDENSATE Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                else:
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (MSm3/Month)')

            else:
                if (cols[n] == 'GAS'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (BSm3/Month)')
                elif (cols[n] == 'GAS Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (BSm3)')
                elif (cols[n] == 'OIL Cumulative'):
                    ax.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'WATER Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'OE Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                elif (cols[n] == 'CONDENSATE Cumulative'):
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production (MSm3)')
                else:
                    ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (MSm3/Month)')

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

    def plot_multi4(self,data, userValues, cols=None, spacing=.05, **kwargs):

        # Get default color style from pandas - can be changed to any other color list
        if cols is None: cols = data.columns
        if len(cols) == 0: return

        # del userValues[-1]
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

        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        # First axis
        ax = data.loc[:, cols[0]].plot(x_compat=True, label=cols[0], color=colors[0], **kwargs)

        # round to nearest years.
        datemin = np.datetime64(data.index[0], 'Y')
        datemax = np.datetime64(list(data.index)[-2], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        ax.grid(axis='both', which='both')

        if (cols[0] == 'GAS'):
            ax.set_ylabel(ylabel=cols[0] + ' Production Rate (BSm3/Month)')
        elif (cols[0] == 'GOR'):
            ax.set_ylabel(ylabel='Gas Oil Ratio (fraction)')
        elif (cols[0] == 'CGR'):
            ax.set_ylabel(ylabel='CONDENSATE GAS Ratio (fraction)')
        elif (cols[0] == 'WCUT'):
            ax.set_ylabel(ylabel='Water Cut (fraction)')
        else:
            ax.set_ylabel(ylabel=cols[0] + ' Production Rate (MSm3/Month)')

        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))

            data.loc[:, cols[n]].plot(ax=ax_new, x_compat=True, label=cols[n], color=colors[n % len(colors)], **kwargs)

            if (cols[n] == 'GAS'):
                ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (BSm3/Month)')
            elif (cols[n] == 'GOR'):
                ax.set_ylabel(ylabel='Gas Oil Ratio (fraction)')
            elif (cols[n] == 'CGR'):
                ax.set_ylabel(ylabel='CONDENSATE GAS Ratio (fraction)')
            elif (cols[n] == 'WCUT'):
                ax.set_ylabel(ylabel='Water Cut (fraction)')
            else:
                ax_new.set_ylabel(ylabel=cols[n] + ' Production Rate (MSm3/Month)')

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
