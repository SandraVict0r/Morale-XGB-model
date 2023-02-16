########################## Libraries #######################################

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
from scipy import stats
from library import *

import statsmodels.stats.multicomp as mc

from IPython.display import HTML, display, Markdown

########################## Data notebook function ##########################

def throw_double_answer(rows_to_keep, df, key) :
    throw_id = []
    df_id = df.groupby("id_participant").count()
    for id in df_id.index :
        if df_id.loc[id][key] > rows_to_keep :
            throw_id.append(id)

    for id in throw_id :    
        rows = df[df.id_participant == id].shape[0]
        last_rows = df[df.id_participant == id].tail(rows-rows_to_keep)
        df = df[~df.isin(last_rows)].dropna(how='all')
    return df

def can_be_converted_to_integer(column):
    try:
        column.astype(int)
        return True
    except ValueError:
        return False

def throw_bad_answers(df,out):
    #collect lines corresponding in data
    index = []
    for i in range(len(out)):
        index.append(df.index[df["id_participant"] == out[i]].tolist())
    #drop lines
    for i in range(len(index)):
        for j in range(len(index[i])):
            df.drop(index[i][j], inplace=True)

########################## Statistical Analysis notebook function ############

def show_pairs_occurence(DF, chars):
    pairs = {}

    for i in chars:
        for j in chars:
            if (i != j) and ((j,i) not  in pairs):
                pairs[(i, j)] = DF[DF.left_char == i][DF.right_char == j].shape[0]
                
    keys = list(pairs.keys())
    values = list(pairs.values())
    plt.bar(["(" + i + "," + j + ")" for (i, j) in keys],
           values,
            width=0.4)
    plt.yticks(size=15)
    plt.xticks(size=15)
    for pair in pairs :
        for i, v in enumerate(pairs.values()):
            plt.text(i ,v + 0.2, str(v), ha='center')
    plt.show()
    sum_pairs = sum(pairs.values())
    mean = sum_pairs // len(pairs) 
    std_dev = statistics.stdev(pairs.values())
    max_key, max_value = max(pairs.items(), key=lambda item: item[1])    
    min_key, min_value = min(pairs.items(), key=lambda item: item[1])
    print("Mean:", mean)
    print("STD:", std_dev)
    print("Max:", max_key, " - ",max_value)
    print("Min:", min_key, " - ",min_value)
    print("Sum:", sum_pairs)
    
def setup_repetition(id_graph, DF, pair, title):
    df_rep = DF[(DF.left_char == pair[0]) & (DF.right_char == pair[1]) & (DF.value_left_rep1 != -1) & (DF.value_left_rep2 != -1)]
    width = {}
    x = range(1,4)
    behavior = [0,0,0]
    for i in range(df_rep.shape[0]):
        #get useful value
        v_l = df_rep.iloc[i].value_left
        v_l_r1 = df_rep.iloc[i].value_left_rep1
        v_l_r2 = df_rep.iloc[i].value_left_rep2
        y = (v_l, v_l_r1, v_l_r2)
        mean_vl = np.mean(y)

        #group the number of sames answers to plot wider lines
        if y in width :
            width[y] += 1
        else :
            width[y] = 1

        #get color o the plot
        if mean_vl > v_l :
            color = 'g'
            label = 'Increase'
            behavior[0] += 1
        elif mean_vl < v_l :
            color = 'r'
            label = 'Decrease'
            behavior[1] += 1
        else :
            color = 'b'
            label = 'Constant'
            behavior[2] += 1

        #plot the line
        id_graph.plot(x,y,linewidth= width[y],color=color,label=label)

    #table of behaviors
    display(Markdown("<table><thead><tr><th># of increases</th><th># of decreases</th><th># of constants</th></tr></thead>" 
                    + "<tbody><tr><td>" + str(behavior[0]) 
                    + "</td><td>" + str(behavior[1]) 
                    + "</td><td>" + str(behavior[2]) 
                    + "</td></tr></tbody></table>"))
    #graph setup
    y_label = []

    id_graph.set_title(title, fontweight='bold', size=15)
    id_graph.set_xticks([1, 2, 3])
    id_graph.set_yticks(range(11))
    id_graph.set_xlabel("Repetition of the question", size=15)
    id_graph.set_ylabel("Gived to " + pair[0], size=15)
    handles, labels = id_graph.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    font = fm.FontProperties(size=15)
    id_graph.legend(by_label.values(),
               by_label.keys(),
               prop=font,
               loc="upper right")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    
def setup_correlation(id_graph, DF, left, right, title, left_label, right_label):
    #set width of bar
    barWidth = 0.25

    #seprate df and put it in array 
    danger = DF[DF.scenario < 4]
    fatigue = DF[(DF.scenario > 3) & (DF.scenario < 7)]
    reward = DF[DF.scenario > 6]
    
    DFS = [danger, fatigue, reward]
    
    #define y and error array
    y = []
    error = []
    anova = []
    
    #fill y and error for each df
    for df in DFS :
        value_left_man = df[df.left_char.isin(left)].value_left
        value_right_woman = 10 - df[df.right_char.isin(right)].value_left
        
        y.append([np.mean(value_left_man),np.mean(value_right_woman)])
        error.append([np.std(value_left_man, ddof=1),np.std(value_right_woman, ddof=1)])
        
        f_value,p_value = stats.f_oneway(value_left_man,value_right_woman)
        f = round(f_value,2)
        p = "{:.2e}".format(p_value)
        anova.append([f,p])
    
    # Set position of bar on X axis
    danger_bar = np.arange(2)
    fatigue_bar = [x + barWidth for x in danger_bar]
    reward_bar = [x + barWidth for x in fatigue_bar]

    # Make the plot
    id_graph.grid()
    id_graph.bar(danger_bar,
            y[0],
            yerr=error[0],
            color='crimson',
            width=barWidth,
            edgecolor='grey',
            label='Danger',
            alpha=0.75,
            ecolor='black',
            capsize=5)
    id_graph.bar(fatigue_bar,
            y[1],
            yerr=error[1],
            color='gold',
            width=barWidth,
            edgecolor='grey',
            label='Fatigue',
            alpha=0.75,
            ecolor='black',
            capsize=5)
    id_graph.bar(reward_bar,
            y[2],
            yerr=error[2],
            color='turquoise',
            width=barWidth,
            edgecolor='grey',
            label='Reward',
            alpha=0.75,
            ecolor='black',
            capsize=5)
    
    #display anova
    show_anova(anova)
    
    #setup graph
    id_graph.set_title(title, fontweight='bold')
    id_graph.set_ylabel('Decision taken')
    id_graph.set_yticks(range(11))
    id_graph.set_xticks([r + barWidth for r in range(2)], [left_label, right_label])
    id_graph.legend()
    
def setup_force_correlation(id_graph, DF, graph_info):
    #set x and y
    x = DF.strength_left
    y = DF.value_left

    #setup graph
    id_graph.scatter(x,y, color= graph_info['color'],s=size_points(DF))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    id_graph.plot(x,p(x),"k--")
    id_graph.set_title(graph_info['title'], fontweight='bold', fontsize=20)
    id_graph.set_ylabel(graph_info['type'] + " given", fontweight='bold', fontsize=15)
    id_graph.set_xlabel('Force perceived', fontweight='bold', fontsize=15)

    #show spearman
    show_spearmann(graph_info['type'],x,y)
    
def size_points(DF):
    dict = {}
    array = []
    for i in DF.index :
        key = (DF.loc[i].strength_left, DF.loc[i].value_left)
        if key in dict.keys():
            dict[key] += 1
        else :
            dict[key] = 1
    
    for i in DF.index :
        x = DF.loc[i].strength_left
        y = DF.loc[i].value_left
        array.append(dict[(x,y)])
    
    return array
    
def show_anova(fp_tab):
    display(Markdown("<table><thead><tr><th></th><th>Danger</th><th>Fatigue</th><th>Reward</th></tr></thead>"
                     + "<tbody><tr><td>f value</td><td>" + str(fp_tab[0][0]) + "</td><td>" + str(fp_tab[1][0]) + "</td><td>" + str(fp_tab[2][0]) + "</td></tr>"
                     +"<tr><td>p value</td><td>" + str(fp_tab[0][1]) + "</td><td>" + str(fp_tab[1][1]) + "</td><td>" + str(fp_tab[2][1]) + "</td></tr></tbody></table>"))
    scenario_type = ["danger","fatigue","reward"]
    for i in range(3) :
        if float(fp_tab[i][1]) > 0.05 :
            display(Markdown("**!!! No significant difference in " + scenario_type[i] + " scenario !!!**"))
            
def show_spearmann(title,x,y):
    coef, p = stats.spearmanr(x, y)
    display(Markdown("### " +title))
    print('Spearmans correlation coefficient: %.3f' % coef)
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=', "{:.2e}".format(p))
    else:
        print('Samples are correlated (reject H0) p=', "{:.2e}".format(p))
        
            
def random_pair():
    a, b = random.sample(range(7), 2)
    while a == b:
        a, b = random.sample(range(7), 2)
    return a,b
