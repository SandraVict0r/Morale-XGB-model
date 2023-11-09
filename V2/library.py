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
import statsmodels.stats.multicomp as mc
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
#from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
from sklearn import tree
from subprocess import call
from tqdm import tqdm
import pickle
from graphviz import Source
import tikzplotlib 

from xgboost import plot_tree, to_graphviz
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models, layers
from keras.layers import *
from keras.models import *
import visualkeras

from sklearn.linear_model import LinearRegression

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

########################## Statistical Analysis for experiment 1 notebook function ############

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
    
def setup_repetition( DF, pair, title):
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
        plt.plot(x,y,linewidth= width[y],color=color,label=label)

    #table of behaviors
    display(Markdown("<table><thead><tr><th># of increases</th><th># of decreases</th><th># of constants</th></tr></thead>" 
                    + "<tbody><tr><td>" + str(behavior[0]) 
                    + "</td><td>" + str(behavior[1]) 
                    + "</td><td>" + str(behavior[2]) 
                    + "</td></tr></tbody></table>"))
    #graph setup
    y_label = []

    plt.set_title(title, fontweight='bold', size=15)
    plt.set_xticks([1, 2, 3])
    plt.set_yticks(range(11))
    plt.set_xlabel("Repetition of the question", size=15)
    plt.set_ylabel("Gived to " + pair[0], size=15)
    handles, labels = id_graph.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    font = fm.FontProperties(size=15)
    plt.legend(by_label.values(),
               by_label.keys(),
               prop=font,
               loc="upper right")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

def bar_plot(df):
    d_l = df[["value_left", "left_char", "scenario"]].rename(
        columns={
            "value_left": "Mean Distribution Attributed",
            "left_char": "Character",
            "scenario": "Scenario",
        }
    )
    d_r = df[["value_right", "right_char", "scenario"]].rename(
        columns={
            "value_right": "Mean Distribution Attributed",
            "right_char": "Character",
            "scenario": "Scenario",
        }
    )
    data = pd.concat(
        [d_l, d_r],
        axis=0,
        ignore_index=True,
    )
    chars = data.Character.unique()
    scenarios = data.Scenario.unique()
    # Add a dark grid
    sns.set(style="darkgrid")
    i = 0
    for scenario in scenarios:
        # Calculer les p-values
        group1 = data[data["Character"] == chars[1]][data["Scenario"] == scenario][
            "Mean Distribution Attributed"
        ]
        group2 = data[data["Character"] == chars[0]][data["Scenario"] == scenario][
            "Mean Distribution Attributed"
        ]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        # Annoter le sns.barplot avec la p-value
        if p_value > 0.005:
            text = f"p-value: {p_value:.3f}"
        else:
            text = "p-value < 0.005"
        # Afficher le résultat sous la forme T(df) = t, p = 0,XXX
        degrees_of_freedom = len(group1) + len(group2) - 2  # Calcul du degré de liberté

        # Formatage du résultat
        result_string = f"T({degrees_of_freedom}) = {t_stat:.3f}, " + text

        plt.text(
            i,
            max(group1.max(), group2.max()) + .5,
            result_string,
            ha="center",
            fontsize=9,
        )
        i += 1

    # Create and display the plot
    sns.barplot(
        data,
        x="Scenario",
        y="Mean Distribution Attributed",
        hue="Character",
        palette="Pastel1",
    )
    plt.ylim([0, 10])
    plt.yticks(np.arange(0, 10, step=1))
    plt.show()
    
    
def setup_correlation(DF, left, right, title, left_label, right_label):
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
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.bar(danger_bar,
            y[0],
            yerr=error[0],
            color='crimson',
            width=barWidth,
            edgecolor='grey',
            label='Risk',
            alpha=0.75,
            ecolor='black',
            capsize=5)
    plt.bar(fatigue_bar,
            y[1],
            yerr=error[1],
            color='gold',
            width=barWidth,
            edgecolor='grey',
            label='Effort',
            alpha=0.75,
            ecolor='black',
            capsize=5)
    plt.bar(reward_bar,
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
    plt.title(title, fontweight='bold')
    plt.ylabel('Mean Decision Attributed', fontweight='bold',)
    plt.yticks(range(11))
    plt.xticks([r + barWidth for r in range(2)], [left_label, right_label])
    plt.legend(["Risk","Effort","Reward"])
    plt.savefig("Figures/" + title + ".png")
    plt.show()

def setup_force_correlation(x,y, graph_info):
    
    #setup graph
    plt.figure(figsize=(10,5))
    plt.scatter(x,y, color= graph_info['color'],s= size_points(x, y))
    slope, intercept = np.polyfit(x, y, 1)
    tendance = "Tendency : {:.2f}x + {:.2f}".format(slope, intercept)
    plt.plot(x,slope*x + intercept,"k--", label=tendance)
    plt.ylabel(graph_info['ylabel'] , fontweight='bold')
    plt.xlabel(graph_info['xlabel'], fontweight='bold')
    plt.xticks(range(min(x),max(x)+1,2))
    plt.yticks(range(min(y),max(y)+1,2))
    plt.xlim(min(x),max(x)+1)
    plt.ylim(min(x),max(y)+1)
    p = show_spearmann(graph_info['type'],x,y)
    if p < 0.0001 :
        p_value = "p < 0.0001"
    else :
        p_value = "p =" + "{:.2e}".format(p)
    # ajout du texte de la tendance de la courbe
    plt.text(10, 2, p_value, fontweight = 'bold', fontsize = 10)
    plt.legend(loc='upper right')
    plt.savefig("Figures/force_corr_" + graph_info['type'] + ".png")
    plt.show()
    
def size_points(x_size, y_size):
    dict = {}
    array = []
    for i in x_size.index :
        key = (x_size.loc[i], y_size.loc[i])
        if key in dict.keys():
            dict[key] += 1
        else :
            dict[key] = 1

    for i in x_size.index :
        x = x_size.loc[i]
        y = y_size.loc[i]
        array.append(dict[(x,y)])
    
    return array
    
    

def show_anova(fp_tab):
    display(Markdown("<table><thead><tr><th></th><th>Risk</th><th>Effort</th><th>Reward</th></tr></thead>"
                     + "<tbody><tr><td>f value</td><td>" + str(fp_tab[0][0]) + "</td><td>" + str(fp_tab[1][0]) + "</td><td>" + str(fp_tab[2][0]) + "</td></tr>"
                     +"<tr><td>p value</td><td>" + str(fp_tab[0][1]) + "</td><td>" + str(fp_tab[1][1]) + "</td><td>" + str(fp_tab[2][1]) + "</td></tr></tbody></table>"))
    scenario_type = ["risk","effort","reward"]
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
    return p
            
def random_pair():
    a, b = random.sample(range(7), 2)
    while a == b:
        a, b = random.sample(range(7), 2)
    return a,b

def sum_absolute_difference(df):
    return abs(df.value_left_rep2 - df.value_left_rep1) + abs(df.value_left - df.value_left_rep1) + abs(df.value_left_rep2 - df.value_left)

def rep_table(df):   
    char_pairs = []
    for i in range(1,8):
        for j in range(i+1,8):
            char_pairs.append(["p"+str(i),"p"+str(j)])

    sad_df = pd.DataFrame(index=['median','std'])

    for pair in char_pairs :
        median = sum_absolute_difference(df[(df.left_char == pair[0]) & (df.right_char == pair[1])]).median()
        std = sum_absolute_difference(df[(df.left_char == pair[0]) & (df.right_char == pair[1])]).std()
        sad_df.insert(len(sad_df.columns), str(pair), [median, round(std,2)])

    return sad_df

########################## Random Forest notebook function ##########################

def classification_report_opti(y_test, y_test_predict):
    classification = {}
    for cl in range(max(y_test) + 1):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(y_test)):
            if cl > 0 and cl < max(y_test):
                if y_test[i] == cl and y_test_predict[i] in [
                        cl - 1, cl, cl + 1
                ]:
                    tp += 1
                elif y_test[i] == cl and y_test_predict[i] not in [
                        cl - 1, cl, cl + 1
                ]:
                    fn += 1
                elif y_test[i] != cl and y_test_predict[i] in [
                        cl - 1, cl, cl + 1
                ]:
                    fp += 1
                elif y_test[i] != cl and y_test_predict[i] not in [
                        cl - 1, cl, cl + 1
                ]:
                    tn += 1
            elif cl == 0:
                if y_test[i] == cl and y_test_predict[i] in [cl, cl + 1]:
                    tp += 1
                elif y_test[i] == cl and y_test_predict[i] not in [cl, cl + 1]:
                    fn += 1
                elif y_test[i] != cl and y_test_predict[i] in [cl, cl + 1]:
                    fp += 1
                elif y_test[i] != cl and y_test_predict[i] not in [cl, cl + 1]:
                    tn += 1
            elif cl == max(y_test):
                if y_test[i] == cl and y_test_predict[i] in [cl, cl - 1]:
                    tp += 1
                elif y_test[i] == cl and y_test_predict[i] not in [cl, cl - 1]:
                    fn += 1
                elif y_test[i] != cl and y_test_predict[i] in [cl, cl - 1]:
                    fp += 1
                elif y_test[i] != cl and y_test_predict[i] not in [cl, cl - 1]:
                    tn += 1
            classification[cl] = {'TN': tn, 'FN': fn, 'FP': fp, 'TP': tp}
    for i in classification:
        print(str(i) + " : ",classification[i]) 
        
    for i in classification:
        precision = (
            classification[i]['TP'] /
            (classification[i]['TP'] + classification[i]['FP'])
        ) if (classification[i]['TP'] + classification[i]['FP']) != 0 else 0
        recall = (
            classification[i]['TP'] /
            (classification[i]['TP'] + classification[i]['FN'])
        ) if (classification[i]['TP'] + classification[i]['FN']) != 0 else 0
        accuracy = (
            (classification[i]['TP'] + classification[i]['TN']) /
            (classification[i]['TP'] + classification[i]['TN'] +
             classification[i]['FP'] + classification[i]['FN'])
        ) if (classification[i]['TP'] + classification[i]['TN'] +
              classification[i]['FP'] + classification[i]['FN']) != 0 else 0
        classification[i].update({
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'accuracy': round(accuracy, 3)
        })
        f1_score = (
            2 * (classification[i]['precision'] * classification[i]['recall'])
        ) / (classification[i]['precision'] + classification[i]['recall']) if (
            classification[i]['precision'] +
            classification[i]['recall']) != 0 else 0
        classification[i].update({'f1-score': round(f1_score, 3)})

    num = 0
    denom = 0
    overall_acc = []
    for i in classification:
        num += classification[i]['TP']
        overall_acc.append(classification[i]['accuracy'])

    accuracy = num / len(y_test_predict)
    classification.update({'accuracy': round(accuracy, 3)})

    print('class | precision | recall | f1-score | accuracy')
    for i in range(max(y_test) + 1):
        print(i, '    | ', classification[i]['precision'], '    | ',
              classification[i]['recall'], ' | ',
              classification[i]['f1-score'], '   | ',
              classification[i]['accuracy'])
    print()
    print('accuracy :    ', classification['accuracy'])
    overall_acc.append(classification['accuracy'])
    return overall_acc

def accuracy_inclasses(y_test, y_predict):
    overall_acc = []
    overall_prec = []
    all_TP = 0
    for i in range(11) :
        conf_matrix = multilabel_confusion_matrix(y_test, y_predict)[i]
        print("class " + str(i) + " : ",conf_matrix)
        TN = conf_matrix[0][0]
        TP = conf_matrix[1][1]
        FN = conf_matrix[1][0]
        FP = conf_matrix[0][1]
        all_TP += TP
        print("TP : ",TP,"TN : ",TN)
        total = TP + TN + FN + FP
        acc = (TP + TN ) / total
        pre = (TP) / (TP + FP)
        overall_acc.append(acc)
        overall_prec.append(pre)
    print(" AVG accuracy :", all_TP/ len(y_test) )
    return overall_acc, overall_prec

########################## Neural Network notebook function ##########################

def plot_curves_confusion (history,confusion_matrix,class_names):
    plt.figure(1,figsize=(16,6))
    plt.gcf().subplots_adjust(left = 0.125, bottom = 0.2, right = 1,
                          top = 0.9, wspace = 0.25, hspace = 0)

    # division de la fenêtre graphique en 1 ligne, 3 colonnes,
    # graphique en position 1 - loss fonction

    plt.subplot(1,3,1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper left')
    # graphique en position 2 - accuracy
    plt.subplot(1,3,2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper left')

    # matrice de correlation
    plt.subplot(1,3,3)
    sns.heatmap(confusion_matrix,annot=True,fmt="d",cmap='Blues',xticklabels=class_names, yticklabels=class_names)# label=class_names)
    # labels, title and ticks
    plt.xlabel('Predicted', fontsize=12)
    #plt.set_label_position('top') 
    #plt.set_ticklabels(class_names, fontsize = 8)
    #plt.tick_top()
    plt.title("Correlation matrix")
    plt.ylabel('True', fontsize=12)
    #plt.set_ticklabels(class_names, fontsize = 8)
    plt.show()

########################## Statistical Analysis for experiment 2 notebook function ############

def get_data_from_questionnaire(quest_ref) :   
    df = pd.read_csv("Datasets/form" + str(quest_ref) + ".csv", sep=";", encoding='latin-1', header=None)
    df = df.drop(range(8), axis = 1)
    df = df.drop(0, axis = 0)
    columns_name = {}
    for col in df.columns :
        columns_name[col] = 'ans' + str(col-7)
    df.rename(columns=columns_name,inplace=True)
    df['questionnaire_reference'] = np.full(len(df), quest_ref)
    return df


def index_classification_per_questions(df,index):
    dataframe = df

    index = sorted(index.items(), key=lambda t: t[1] == "CH")
    index = sorted(index, key=lambda t: t[1] == "3RH")
    index = sorted(index, key=lambda t: t[1] == "2RH")
    index = sorted(index, key=lambda t: t[1] == "CR")
    index = sorted(index, key=lambda t: t[1] == "3RR")
    index = sorted(index, key=lambda t: t[1] == "2RR")
    index = sorted(index, key=lambda t: t[1] == "CI")

    columns = np.array([])
    for i in range(22) :
        columns = np.append(columns, index[i][0])
    dataframe = dataframe.reindex(columns = columns)
    dataframe = dataframe.rename(columns = dict(index), errors='raise')
    return dataframe

def set_label_per_question(df):
    label = {"human" : ["CH","3RH","2RH"],
                "robot" : ["CR","3RR","2RR"],
                "inversed" : ["CI"]}
    columns = {}
    for i in range(len(df.columns)-1):
        if df.columns[i] in label["human"] :
            columns[df.columns[i]] = "human"
        elif df.columns[i] in label["robot"] :
            columns[df.columns[i]] = "robot"
        elif df.columns[i] in label["inversed"] :
            columns[df.columns[i]] = "inversed"

    dataframe = df.rename(columns=columns,
                        errors='raise')
    return dataframe


def get_Y(string,df) :
    df_col = df[string]
    if string == 'inversed' :
         df_col.columns = ['q1', 'q2', 'q3']
    else :
        df_col.columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']
    df_X = pd.DataFrame()
    for i in range(len(df_col.columns)):
        df_X = pd.concat([df_X,df_col['q'+str(i+1)]], ignore_index=True)
    df_X.columns = [string]
    nb_X = np.arange(11)
    for i in range(11):
        nb_X[i] = df_X[df_X[string] == i].count().values
    nb_X = (nb_X/df_X.count().values)*100
    return nb_X


def koglomorov_test(data1, data2) :
    # Calculez le test de Koglomorov-Smirnov entre les deux courbes
    statistic, p_value = ks_2samp(data1,data2)

    # Imprimez le résultat du test
    print(f"Statistique de Koglomorov-Smirnov : {statistic:.5f}")
    print(f"P-valeur : {p_value:.10f}")

    # Interprétez le résultat
    if p_value < 0.05:
        print("Il y a une différence significative entre les deux courbes.")
    else:
        print("Il n'y a pas de différence significative entre les deux courbes.")